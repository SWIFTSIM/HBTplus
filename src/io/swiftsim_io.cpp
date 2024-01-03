using namespace std;
#include <iostream>
#include <numeric>
// #include <iomanip>
#include <assert.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <list>
#include <sstream>
#include <string>
#include <typeinfo>

#include "../config_parser.h"
#include "../hdf_wrapper.h"
#include "../mymath.h"
#include "../snapshot.h"
#include "swiftsim_io.h"
#include "exchange_and_merge.h"

void create_SwiftSimHeader_MPI_type(MPI_Datatype &dtype)
{
  /*to create the struct data type for communication*/
  SwiftSimHeader_t p;
#define NumAttr 14
  MPI_Datatype oldtypes[NumAttr];
  int blockcounts[NumAttr];
  MPI_Aint offsets[NumAttr], origin, extent;

  MPI_Get_address(&p, &origin);
  MPI_Get_address((&p) + 1, &extent); // to get the extent of s
  extent -= origin;

  int i = 0;
#define RegisterAttr(x, type, count)                                                                                   \
  {                                                                                                                    \
    MPI_Get_address(&(p.x), offsets + i);                                                                              \
    offsets[i] -= origin;                                                                                              \
    oldtypes[i] = type;                                                                                                \
    blockcounts[i] = count;                                                                                            \
    i++;                                                                                                               \
  }
  RegisterAttr(NumberOfFiles, MPI_INT, 1) RegisterAttr(BoxSize, MPI_DOUBLE, 1) RegisterAttr(ScaleFactor, MPI_DOUBLE, 1)
    RegisterAttr(OmegaM0, MPI_DOUBLE, 1) RegisterAttr(OmegaLambda0, MPI_DOUBLE, 1) RegisterAttr(h, MPI_DOUBLE, 1)
      RegisterAttr(mass, MPI_DOUBLE, TypeMax) RegisterAttr(npart[0], MPI_INT, TypeMax)
        RegisterAttr(npartTotal[0], MPI_HBT_INT, TypeMax) RegisterAttr(length_conversion, MPI_DOUBLE, 1)
          RegisterAttr(mass_conversion, MPI_DOUBLE, 1) RegisterAttr(velocity_conversion, MPI_DOUBLE, 1)
            RegisterAttr(energy_conversion, MPI_DOUBLE, 1) RegisterAttr(NullGroupId, MPI_INTEGER, 1)

#undef RegisterAttr
              assert(i <= NumAttr);

  MPI_Type_create_struct(i, blockcounts, offsets, oldtypes, &dtype);
  MPI_Type_create_resized(dtype, (MPI_Aint)0, extent, &dtype);
  MPI_Type_commit(&dtype);
#undef NumAttr
}

void SwiftSimReader_t::SetSnapshot(int snapshotId)
{
  if (HBTConfig.SnapshotNameList.empty())
  {
    stringstream formatter;
    if (HBTConfig.SnapshotDirBase.length() > 0)
      formatter << HBTConfig.SnapshotDirBase << "_" << setw(4) << setfill('0') << snapshotId << "/";
    formatter << HBTConfig.SnapshotFileBase << "_" << setw(4) << setfill('0') << snapshotId;
    SnapshotName = formatter.str();
  }
  else
    SnapshotName = HBTConfig.SnapshotNameList[snapshotId];
}

void SwiftSimReader_t::GetFileName(int ifile, string &filename)
{
  stringstream formatter;
  if (ifile < 0)
    formatter << HBTConfig.SnapshotPath << "/" << SnapshotName << ".hdf5";
  else
    formatter << HBTConfig.SnapshotPath << "/" << SnapshotName << "." << ifile << ".hdf5";
  filename = formatter.str();
}

hid_t SwiftSimReader_t::OpenFile(int ifile)
{
  string filename;

  H5E_auto_t err_func;
  char *err_data;
  H5Eget_auto(H5E_DEFAULT, &err_func, (void **)&err_data);
  H5Eset_auto(H5E_DEFAULT, NULL, NULL);

  /* Try filename with index first (e.g. snap_0001.0.hdf5) */
  GetFileName(ifile, filename);
  hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

  /* If that failed, try without an index (e.g. snap_0001.hdf5),
     but only if we're reading file 0 */
  if (file < 0 && ifile == 0)
  {
    GetFileName(-1, filename);
    file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  }

  if (file < 0)
  {
    cout << "Failed to open file: " << filename << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  H5Eset_auto(H5E_DEFAULT, err_func, (void *)err_data);

  return file;
}

void SwiftSimReader_t::ReadHeader(int ifile, SwiftSimHeader_t &header)
{
  double BoxSize_3D[3];
  int NumPartTypes;

  hid_t file = OpenFile(ifile);
  ReadAttribute(file, "Header", "NumPartTypes", H5T_NATIVE_INT, &NumPartTypes);
  ReadAttribute(file, "Header", "NumFilesPerSnapshot", H5T_NATIVE_INT, &Header.NumberOfFiles);
  ReadAttribute(file, "Header", "BoxSize", H5T_NATIVE_DOUBLE, BoxSize_3D);
  if (BoxSize_3D[0] != BoxSize_3D[1] || BoxSize_3D[0] != BoxSize_3D[2])
  {
    cout << "Swift simulation box must have equal size in each dimension!\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  Header.BoxSize = BoxSize_3D[0]; // Can only handle cubic boxes
  ReadAttribute(file, "Cosmology", "Scale-factor", H5T_NATIVE_DOUBLE, &Header.ScaleFactor);
  ReadAttribute(file, "Cosmology", "Omega_m", H5T_NATIVE_DOUBLE, &Header.OmegaM0);
  ReadAttribute(file, "Cosmology", "Omega_lambda", H5T_NATIVE_DOUBLE, &Header.OmegaLambda0);
  ReadAttribute(file, "Cosmology", "h", H5T_NATIVE_DOUBLE, &Header.h);
  for (int i = 0; i < TypeMax; i += 1)
    Header.mass[i] = 0.0; // Swift particles always have individual masses

  /* Read physical constants assumed by SWIFT */
  double parsec_cgs;
  ReadAttribute(file, "PhysicalConstants/CGS", "parsec", H5T_NATIVE_DOUBLE, &parsec_cgs);
  double solar_mass_cgs;
  ReadAttribute(file, "PhysicalConstants/CGS", "solar_mass", H5T_NATIVE_DOUBLE, &solar_mass_cgs);
  const double km_cgs = 1.0e5;

  /* Read unit system used by SWIFT */
  double length_cgs;
  ReadAttribute(file, "Units", "Unit length in cgs (U_L)", H5T_NATIVE_DOUBLE, &length_cgs);
  double mass_cgs;
  ReadAttribute(file, "Units", "Unit mass in cgs (U_M)", H5T_NATIVE_DOUBLE, &mass_cgs);
  double time_cgs;
  ReadAttribute(file, "Units", "Unit time in cgs (U_t)", H5T_NATIVE_DOUBLE, &time_cgs);

  /* Read group ID used to indicate that a particle is in no FoF group */
  string buf;
  ReadAttribute(file, "Parameters", "FOF:group_id_default", buf);
  Header.NullGroupId = std::stoi(buf);

  /* Compute conversion from SWIFT's unit system to HBT's unit system (apart from any a factors) */
  Header.length_conversion = (length_cgs / (1.0e6 * parsec_cgs)) * Header.h / HBTConfig.LengthInMpch;
  Header.mass_conversion = (mass_cgs / solar_mass_cgs) * Header.h / HBTConfig.MassInMsunh;
  Header.velocity_conversion = (length_cgs / time_cgs) / km_cgs / HBTConfig.VelInKmS;
  Header.energy_conversion = Header.velocity_conversion * Header.velocity_conversion;

  /* Convert box size to HBT units */
  Header.BoxSize *= Header.length_conversion;

  /*
     Read per-type header entries

     There may be more than TypeMax particle types. Here we ignore
     any extra types (e.g. neutrinos). The number of types we use
     is the smaller of Swift's NumPartTypes and HBT's TypeMax.
  */
  int NumPart_ThisFile[NumPartTypes]; // same size as attributes in the file
  ReadAttribute(file, "Header", "NumPart_ThisFile", H5T_NATIVE_INT, NumPart_ThisFile);
  unsigned int NumPart_Total[NumPartTypes];
  ReadAttribute(file, "Header", "NumPart_Total", H5T_NATIVE_UINT, NumPart_Total);
  unsigned int NumPart_Total_HighWord[NumPartTypes];
  ReadAttribute(file, "Header", "NumPart_Total_HighWord", H5T_NATIVE_UINT, NumPart_Total_HighWord);
  for (int i = 0; i < TypeMax; i++)
    if (i < NumPartTypes)
      Header.npartTotal[i] = (((unsigned long)NumPart_Total_HighWord[i]) << 32) | NumPart_Total[i];
    else
      Header.npartTotal[i] = 0;

  /* Read softening values used by SWIFT and convert them to internal HBT units */
  // NOTE: Whenever reading "Parameters", we need to use a string to load into.

  // DM softening
  ReadAttribute(file, "Parameters", "Gravity:comoving_DM_softening", buf);
  Header.DM_comoving_softening = stof(buf) * Header.length_conversion;
  ReadAttribute(file, "Parameters", "Gravity:max_physical_DM_softening", buf);
  Header.DM_maximum_physical_softening = stof(buf) * Header.length_conversion;

  // Baryonic softening
  ReadAttribute(file, "Parameters", "Gravity:comoving_baryon_softening", buf);
  Header.baryon_comoving_softening = stof(buf) * Header.length_conversion;
  ReadAttribute(file, "Parameters", "Gravity:max_physical_baryon_softening", buf);
  Header.baryon_maximum_physical_softening = stof(buf) * Header.length_conversion;

  H5Fclose(file);
}

void SwiftSimReader_t::GetParticleCountInFile(hid_t file, int np[])
{
  int NumPartTypes;
  ReadAttribute(file, "Header", "NumPartTypes", H5T_NATIVE_INT, &NumPartTypes);
  int NumPart_ThisFile[NumPartTypes]; // same size as attributes in the file
  ReadAttribute(file, "Header", "NumPart_ThisFile", H5T_NATIVE_INT, NumPart_ThisFile);
  for (int i = 0; i < TypeMax; i++)
    if (i < NumPartTypes)
      np[i] = NumPart_ThisFile[i];
    else
      np[i] = 0;
#ifdef DM_ONLY
  for (int i = 0; i < TypeMax; i++)
    if (i != TypeDM)
      np[i] = 0;
#endif
}

HBTInt SwiftSimReader_t::CompileFileOffsets(int nfiles)
{
  HBTInt offset = 0;
  np_file.reserve(nfiles);
  offset_file.reserve(nfiles);
  for (int ifile = 0; ifile < nfiles; ifile++)
  {
    offset_file.push_back(offset);

    int np_this[TypeMax];
    hid_t file = OpenFile(ifile);
    GetParticleCountInFile(file, np_this);
    H5Fclose(file);
    HBTInt np = accumulate(begin(np_this), end(np_this), (HBTInt)0);

    np_file.push_back(np);
    offset += np;
  }
  return offset;
}

static void check_id_size(hid_t loc)
{
  hid_t dset = H5Dopen2(loc, "ParticleIDs", H5P_DEFAULT);
  hid_t dtype = H5Dget_type(dset);
  size_t ParticleIDStorageSize = H5Tget_size(dtype);
  assert(sizeof(HBTInt) >= ParticleIDStorageSize); // use HBTi8 or HBTdouble if you need long int for id
  H5Tclose(dtype);
  H5Dclose(dset);
}

void SwiftSimReader_t::ReadSnapshot(int ifile, Particle_t *ParticlesInFile)
{
  hid_t file = OpenFile(ifile);
  vector<int> np_this(TypeMax);
  vector<HBTInt> offset_this(TypeMax);
  GetParticleCountInFile(file, np_this.data());
  CompileOffsets(np_this, offset_this);

  HBTReal boxsize = Header.BoxSize;
  for (int itype = 0; itype < TypeMax; itype++)
  {
    int np = np_this[itype];
    if (np == 0)
      continue;
    auto ParticlesThisType = ParticlesInFile + offset_this[itype];
    stringstream grpname;
    grpname << "PartType" << itype;
    const std::string &tmp = grpname.str();
    const char *group_name = tmp.c_str();
    if (!H5Lexists(file, group_name, H5P_DEFAULT))
      continue;

    hid_t particle_data = H5Gopen2(file, grpname.str().c_str(), H5P_DEFAULT);
    check_id_size(particle_data);

    const hsize_t chunksize = 10 * 1024 * 1024;
    const hsize_t nr = (hsize_t)np;

    // Positions
    {
      // Check that positions are comoving
      HBTReal aexp;
      ReadAttribute(particle_data, "Coordinates", "a-scale exponent", H5T_HBTReal, &aexp);
      if (aexp != 1.0)
      {
        cout << "Can't handle Coordinates with a-scale exponent != 1\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
      }

      // Read data in chunks to minimize memory overhead
      for (hsize_t offset = 0; offset < nr; offset += chunksize)
      {
        // Read the next chunk
        hsize_t count = nr - offset;
        if (count > chunksize)
          count = chunksize;
        vector<HBTxyz> x(count);
        ReadPartialDataset(particle_data, "Coordinates", H5T_HBTReal, x.data(), offset, count);
        // Convert to HBT units
        for (hsize_t i = 0; i < count; i++)
          for (int j = 0; j < 3; j++)
            x[i][j] *= Header.length_conversion;
        // Box wrap if necessary
        if (HBTConfig.PeriodicBoundaryOn)
        {
          for (hsize_t i = 0; i < count; i++)
            for (int j = 0; j < 3; j++)
              x[i][j] = position_modulus(x[i][j], boxsize);
        }
        // Store the particle positions
        for (hsize_t i = 0; i < count; i += 1)
          for (int j = 0; j < 3; j += 1)
            ParticlesThisType[offset + i].ComovingPosition[j] = x[i][j];
      }
    }

    // Velocities
    {
      HBTReal aexp;
      ReadAttribute(particle_data, "Velocities", "a-scale exponent", H5T_HBTReal, &aexp);

      // Read data in chunks to minimize memory overhead
      for (hsize_t offset = 0; offset < nr; offset += chunksize)
      {
        // Read the next chunk
        hsize_t count = nr - offset;
        if (count > chunksize)
          count = chunksize;
        vector<HBTxyz> v(count);
        ReadPartialDataset(particle_data, "Velocities", H5T_HBTReal, v.data(), offset, count);
        // Convert units and store the particle velocities
        for (hsize_t i = 0; i < count; i += 1)
          for (int j = 0; j < 3; j += 1)
            ParticlesThisType[offset + i].PhysicalVelocity[j] =
              v[i][j] * Header.velocity_conversion * pow(Header.ScaleFactor, aexp);
      }
    }

    // Ids
    {
      for (hsize_t offset = 0; offset < nr; offset += chunksize)
      {
        hsize_t count = nr - offset;
        if (count > chunksize)
          count = chunksize;
        vector<HBTInt> id(count);
        ReadPartialDataset(particle_data, "ParticleIDs", H5T_HBTInt, id.data(), offset, count);
        for (hsize_t i = 0; i < count; i += 1)
          ParticlesThisType[offset + i].Id = id[i];
      }
    }

    // Masses
    {
      HBTReal aexp;
      std::string name;
      if (itype == 5)
        name = "DynamicalMasses";
      else
        name = "Masses";
      ReadAttribute(particle_data, name.c_str(), "a-scale exponent", H5T_HBTReal, &aexp);
      for (hsize_t offset = 0; offset < nr; offset += chunksize)
      {
        hsize_t count = nr - offset;
        if (count > chunksize)
          count = chunksize;
        vector<HBTReal> m(count);
        ReadPartialDataset(particle_data, name.c_str(), H5T_HBTReal, m.data(), offset, count);
        for (hsize_t i = 0; i < count; i += 1)
          ParticlesThisType[offset + i].Mass = m[i] * Header.mass_conversion * pow(Header.ScaleFactor, aexp);
      }
    }

#ifndef DM_ONLY
    // internal energy
#ifdef HAS_THERMAL_ENERGY
    if (itype == 0)
    {
      cout << "Reading internal energy from SWIFT not implemented yet!\n";
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
#endif
    { // type
      ParticleType_t t = static_cast<ParticleType_t>(itype);
      for (int i = 0; i < np; i++)
        ParticlesThisType[i].Type = t;
    }
#endif
    H5Gclose(particle_data);
  }
  H5Fclose(file);
}

void SwiftSimReader_t::ReadGroupParticles(int ifile, SwiftParticleHost_t *ParticlesInFile, bool FlagReadParticleId)
{
  hid_t file = OpenFile(ifile);
  vector<int> np_this(TypeMax);
  vector<HBTInt> offset_this(TypeMax);
  GetParticleCountInFile(file, np_this.data());
  CompileOffsets(np_this, offset_this);

  HBTReal boxsize = Header.BoxSize;
  for (int itype = 0; itype < TypeMax; itype++)
  {
    int np = np_this[itype];
    if (np == 0)
      continue;
    auto ParticlesThisType = ParticlesInFile + offset_this[itype];
    stringstream grpname;
    grpname << "PartType" << itype;
    if (!H5Lexists(file, grpname.str().c_str(), H5P_DEFAULT))
      continue;
    hid_t particle_data = H5Gopen2(file, grpname.str().c_str(), H5P_DEFAULT);

    const hsize_t chunksize = 10 * 1024 * 1024;
    const hsize_t nr = (hsize_t)np;

    if (FlagReadParticleId)
    {

      // Positions
      {
        // Check that positions are comoving
        HBTReal aexp;
        ReadAttribute(particle_data, "Coordinates", "a-scale exponent", H5T_HBTReal, &aexp);
        if (aexp != 1.0)
        {
          cout << "Can't handle Coordinates with a-scale exponent != 1\n";
          MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Read data in chunks to minimize memory overhead
        for (hsize_t offset = 0; offset < nr; offset += chunksize)
        {
          // Read the next chunk
          hsize_t count = nr - offset;
          if (count > chunksize)
            count = chunksize;
          vector<HBTxyz> x(count);
          ReadPartialDataset(particle_data, "Coordinates", H5T_HBTReal, x.data(), offset, count);
          // Convert to HBT units
          for (hsize_t i = 0; i < count; i++)
            for (int j = 0; j < 3; j++)
              x[i][j] *= Header.length_conversion;
          // Box wrap if necessary
          if (HBTConfig.PeriodicBoundaryOn)
          {
            for (hsize_t i = 0; i < count; i++)
              for (int j = 0; j < 3; j++)
                x[i][j] = position_modulus(x[i][j], boxsize);
          }
          // Store the particle positions
          for (hsize_t i = 0; i < count; i += 1)
            for (int j = 0; j < 3; j += 1)
              ParticlesThisType[offset + i].ComovingPosition[j] = x[i][j];
        }
      }

      // Velocities
      {
        HBTReal aexp;
        ReadAttribute(particle_data, "Velocities", "a-scale exponent", H5T_HBTReal, &aexp);

        // Read data in chunks to minimize memory overhead
        for (hsize_t offset = 0; offset < nr; offset += chunksize)
        {
          // Read the next chunk
          hsize_t count = nr - offset;
          if (count > chunksize)
            count = chunksize;
          vector<HBTxyz> v(count);
          ReadPartialDataset(particle_data, "Velocities", H5T_HBTReal, v.data(), offset, count);
          // Convert units and store the particle velocities
          for (hsize_t i = 0; i < count; i += 1)
            for (int j = 0; j < 3; j += 1)
              ParticlesThisType[offset + i].PhysicalVelocity[j] =
                v[i][j] * Header.velocity_conversion * pow(Header.ScaleFactor, aexp);
        }
      }

      // Ids
      {
        for (hsize_t offset = 0; offset < nr; offset += chunksize)
        {
          hsize_t count = nr - offset;
          if (count > chunksize)
            count = chunksize;
          vector<HBTInt> id(count);
          ReadPartialDataset(particle_data, "ParticleIDs", H5T_HBTInt, id.data(), offset, count);
          for (hsize_t i = 0; i < count; i += 1)
            ParticlesThisType[offset + i].Id = id[i];
        }
      }

      // Masses
      {
        HBTReal aexp;
        std::string name;
        if (itype == 5)
          name = "DynamicalMasses";
        else
          name = "Masses";
        ReadAttribute(particle_data, name.c_str(), "a-scale exponent", H5T_HBTReal, &aexp);
        for (hsize_t offset = 0; offset < nr; offset += chunksize)
        {
          hsize_t count = nr - offset;
          if (count > chunksize)
            count = chunksize;
          vector<HBTReal> m(count);
          ReadPartialDataset(particle_data, name.c_str(), H5T_HBTReal, m.data(), offset, count);
          for (hsize_t i = 0; i < count; i += 1)
            ParticlesThisType[offset + i].Mass = m[i] * Header.mass_conversion * pow(Header.ScaleFactor, aexp);
        }
      }

#ifndef DM_ONLY
      // internal energy
#ifdef HAS_THERMAL_ENERGY
      if (itype == 0)
      {
        cout << "Reading internal energy from SWIFT not implemented yet!\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
#endif
      { // type
        ParticleType_t t = static_cast<ParticleType_t>(itype);
        for (int i = 0; i < np; i++)
          ParticlesThisType[i].Type = t;
      }
#endif
    }

    // Hostid
    {
      for (hsize_t offset = 0; offset < nr; offset += chunksize)
      {
        hsize_t count = nr - offset;
        if (count > chunksize)
          count = chunksize;
        vector<HBTInt> id(count);
        ReadPartialDataset(particle_data, "FOFGroupIDs", H5T_HBTInt, id.data(), offset, count);
        for (hsize_t i = 0; i < count; i += 1)
          ParticlesThisType[offset + i].HostId =
            (id[i] < 0 ? Header.NullGroupId : id[i]); // negative means outside fof but within Rv
      }
    }
    H5Gclose(particle_data);
  }
  H5Fclose(file);
}

void SwiftSimReader_t::LoadSnapshot(MpiWorker_t &world, int snapshotId, vector<Particle_t> &Particles,
                                    Cosmology_t &Cosmology)
{

  MPI_Barrier(world.Communicator);

  SetSnapshot(snapshotId);

  const int root = 0;
  if (world.rank() == root)
  {
    ReadHeader(0, Header);
    CompileFileOffsets(Header.NumberOfFiles);

    /* Report conversion factors used to go from SWIFT to HBT units */
    cout << "Conversion factor from SWIFT length units to " << HBTConfig.LengthInMpch
         << " Mpc/h = " << Header.length_conversion << endl;
    cout << "Conversion factor from SWIFT mass units to " << HBTConfig.MassInMsunh
         << " Msun/h = " << Header.mass_conversion << endl;
    cout << "Conversion factor from SWIFT velocity units to " << HBTConfig.VelInKmS
         << " km/s = " << Header.velocity_conversion << endl;
    cout << "Null group ID is " << Header.NullGroupId << endl;
  }
  MPI_Bcast(&Header, 1, MPI_SwiftSimHeader_t, root, world.Communicator);
  world.SyncContainer(np_file, MPI_HBT_INT, root);
  world.SyncContainer(offset_file, MPI_HBT_INT, root);

  Cosmology.Set(Header.ScaleFactor, Header.OmegaM0, Header.OmegaLambda0);

  /* Use the softening values we read in from the Header */
  HBTConfig.SofteningHalo = Header.DM_comoving_softening;
  HBTConfig.MaxPhysicalSofteningHalo = Header.DM_maximum_physical_softening;

  HBTInt nfiles_skip, nfiles_end;
  AssignTasks(world.rank(), world.size(), Header.NumberOfFiles, nfiles_skip, nfiles_end);
  {
    HBTInt np = 0;
    np = accumulate(np_file.begin() + nfiles_skip, np_file.begin() + nfiles_end, np);
    Particles.resize(np);
  }

  for (int i = 0, ireader = 0; i < world.size(); i++, ireader++)
  {
    if (ireader == HBTConfig.MaxConcurrentIO)
    {
      ireader = 0;                     // reset reader count
      MPI_Barrier(world.Communicator); // wait for every thread to arrive.
    }
    if (i == world.rank()) // read
    {
      for (int iFile = nfiles_skip; iFile < nfiles_end; iFile++)
      {
        ReadSnapshot(iFile, Particles.data() + offset_file[iFile] - offset_file[nfiles_skip]);
      }
    }
  }

  MPI_Barrier(world.Communicator);
}

inline bool CompParticleHost(const SwiftParticleHost_t &a, const SwiftParticleHost_t &b)
{
  return a.HostId < b.HostId;
}

void SwiftSimReader_t::LoadGroups(MpiWorker_t &world, int snapshotId, vector<Halo_t> &Halos)
{ // read in particle properties at the same time, to avoid particle look-up at later stage.
  SetSnapshot(snapshotId);

  const int root = 0;
  if (world.rank() == root)
  {
    ReadHeader(0, Header);
    CompileFileOffsets(Header.NumberOfFiles);
  }
  MPI_Bcast(&Header, 1, MPI_SwiftSimHeader_t, root, world.Communicator);
  world.SyncContainer(np_file, MPI_HBT_INT, root);
  world.SyncContainer(offset_file, MPI_HBT_INT, root);

  vector<SwiftParticleHost_t> ParticleHosts;
  HBTInt nfiles_skip, nfiles_end;
  AssignTasks(world.rank(), world.size(), Header.NumberOfFiles, nfiles_skip, nfiles_end);
  {
    HBTInt np = 0;
    np = accumulate(np_file.begin() + nfiles_skip, np_file.begin() + nfiles_end, np);
    ParticleHosts.resize(np);
  }
  bool FlagReadId = true; //! HBTConfig.GroupLoadedIndex;

  for (int i = 0, ireader = 0; i < world.size(); i++, ireader++)
  {
    if (ireader == HBTConfig.MaxConcurrentIO)
    {
      ireader = 0;                     // reset reader count
      MPI_Barrier(world.Communicator); // wait for every thread to arrive.
    }
    if (i == world.rank()) // read
    {
      for (int iFile = nfiles_skip; iFile < nfiles_end; iFile++)
        ReadGroupParticles(iFile, ParticleHosts.data() + offset_file[iFile] - offset_file[nfiles_skip], FlagReadId);
    }
  }

  sort(ParticleHosts.begin(), ParticleHosts.end(), CompParticleHost);
  if (!ParticleHosts.empty())
  {
    assert(ParticleHosts.back().HostId <= Header.NullGroupId); // max haloid==NullGroupId
    assert(ParticleHosts.front().HostId >= 0);                 // min haloid>=0
  }

  struct HaloLen_t
  {
    HBTInt haloid;
    HBTInt np;
    HaloLen_t(){};
    HaloLen_t(HBTInt haloid, HBTInt np) : haloid(haloid), np(np)
    {
    }
  };
  vector<HaloLen_t> HaloLen;

  HBTInt curr_host_id = Header.NullGroupId;
  for (auto &&p : ParticleHosts)
  {
    if (p.HostId == Header.NullGroupId)
      break; // NullGroupId comes last
    if (p.HostId != curr_host_id)
    {
      curr_host_id = p.HostId;
      HaloLen.emplace_back(curr_host_id, 1);
    }
    else
      HaloLen.back().np++;
  }
  Halos.resize(HaloLen.size());
  for (HBTInt i = 0; i < Halos.size(); i++)
  {
    Halos[i].HaloId = HaloLen[i].haloid;
    Halos[i].Particles.resize(HaloLen[i].np);
  }
  auto p_in = ParticleHosts.begin();
  for (auto &&h : Halos)
  {
    for (auto &&p : h.Particles)
    {
      p = *p_in;
      ++p_in;
    }
  }

  VectorFree(ParticleHosts);

  ExchangeAndMerge(world, Halos);

  //   cout<<Halos.size()<<" groups loaded";
  //   if(Halos.size()) cout<<" : "<<Halos[0].Particles.size();
  //   if(Halos.size()>1) cout<<","<<Halos[1].Particles.size()<<"...";
  //   cout<<endl;

  //   HBTInt np=0;
  //   for(auto &&h: Halos)
  //     np+=h.Particles.size();
  //   MPI_Allreduce(MPI_IN_PLACE, &np, 1, MPI_HBT_INT, MPI_SUM, world.Communicator);
  //   return np;

  HBTConfig.GroupLoadedFullParticle = true;
}

bool IsSwiftSimGroup(const string &GroupFileFormat)
{
  return GroupFileFormat.substr(0, 8) == "swiftsim";
}
