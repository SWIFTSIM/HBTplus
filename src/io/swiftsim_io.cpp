using namespace std;
#include <iostream>
#include <numeric>
#include <cassert>
// #include <iomanip>
#include <sstream>
#include <string>
#include <typeinfo>
#include <assert.h>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <list>

#include "../snapshot.h"
#include "../mymath.h"
#include "../hdf_wrapper.h"
#include "swiftsim_io.h"
#include "exchange_and_merge.h"
#include "../config_parser.h"

void create_SwiftSimHeader_MPI_type(MPI_Datatype& dtype)
{
  /*to create the struct data type for communication*/	
  SwiftSimHeader_t p;
  #define NumAttr 14
  MPI_Datatype oldtypes[NumAttr];
  int blockcounts[NumAttr];
  MPI_Aint   offsets[NumAttr], origin,extent;
  
  MPI_Get_address(&p,&origin);
  MPI_Get_address((&p)+1,&extent);//to get the extent of s
  extent-=origin;
  
  int i=0;
  #define RegisterAttr(x, type, count) {MPI_Get_address(&(p.x), offsets+i); offsets[i]-=origin; oldtypes[i]=type; blockcounts[i]=count; i++;}
  RegisterAttr(NumberOfFiles, MPI_INT, 1)
  RegisterAttr(BoxSize, MPI_DOUBLE, 1)
  RegisterAttr(ScaleFactor, MPI_DOUBLE, 1)
  RegisterAttr(OmegaM0, MPI_DOUBLE, 1)
  RegisterAttr(OmegaLambda0, MPI_DOUBLE, 1)
  RegisterAttr(h, MPI_DOUBLE, 1)
  RegisterAttr(mass, MPI_DOUBLE, TypeMax)
  RegisterAttr(npart[0], MPI_INT, TypeMax)
  RegisterAttr(npartTotal[0], MPI_HBT_INT, TypeMax)
  RegisterAttr(length_conversion, MPI_DOUBLE, 1)
  RegisterAttr(mass_conversion, MPI_DOUBLE, 1)
  RegisterAttr(velocity_conversion, MPI_DOUBLE, 1)
  RegisterAttr(energy_conversion, MPI_DOUBLE, 1)
  RegisterAttr(NullGroupId, MPI_INTEGER, 1)
    
  #undef RegisterAttr
  assert(i<=NumAttr);
  
  MPI_Type_create_struct(i,blockcounts,offsets,oldtypes, &dtype);
  MPI_Type_create_resized(dtype,(MPI_Aint)0, extent, &dtype);
  MPI_Type_commit(&dtype);
  #undef NumAttr
}

void SwiftSimReader_t::SetSnapshot(int snapshotId)
{  
  if(HBTConfig.SnapshotNameList.empty())
  {
	stringstream formatter;
        if(HBTConfig.SnapshotDirBase.length()>0)
          formatter<<HBTConfig.SnapshotDirBase<<"_"<<setw(4)<<setfill('0')<<snapshotId<<"/";
	formatter<<HBTConfig.SnapshotFileBase<<"_"<<setw(4)<<setfill('0')<<snapshotId;
	SnapshotName=formatter.str();
  }
  else
	SnapshotName=HBTConfig.SnapshotNameList[snapshotId];
}

void SwiftSimReader_t::GetFileName(int ifile, string &filename)
{
  stringstream formatter;
  if(ifile < 0)
    formatter<<HBTConfig.SnapshotPath<<"/"<<SnapshotName<<".hdf5";
  else
    formatter<<HBTConfig.SnapshotPath<<"/"<<SnapshotName<<"."<<ifile<<".hdf5";
  filename=formatter.str();
}

hid_t SwiftSimReader_t::OpenFile(int ifile)
{
  string filename;

  H5E_auto_t err_func;
  char *err_data;
  H5Eget_auto(H5E_DEFAULT, &err_func, (void **) &err_data); 
  H5Eset_auto(H5E_DEFAULT, NULL, NULL);

  /* Try filename with index first (e.g. snap_0001.0.hdf5) */
  GetFileName(ifile, filename);
  hid_t file=H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

  /* If that failed, try without an index (e.g. snap_0001.hdf5),
     but only if we're reading file 0 */
  if(file < 0 && ifile==0)
  {
    GetFileName(-1, filename);
    file=H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT); 
  }

  if(file < 0) {
    cout << "Failed to open file: " << filename << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  H5Eset_auto(H5E_DEFAULT, err_func, (void *) err_data); 

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
  if(BoxSize_3D[0]!=BoxSize_3D[1] || BoxSize_3D[0]!=BoxSize_3D[2]) {
    cout << "Swift simulation box must have equal size in each dimension!\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  Header.BoxSize = BoxSize_3D[0]; // Can only handle cubic boxes
  ReadAttribute(file, "Cosmology", "Scale-factor", H5T_NATIVE_DOUBLE, &Header.ScaleFactor);
  ReadAttribute(file, "Cosmology", "Omega_m", H5T_NATIVE_DOUBLE, &Header.OmegaM0);
  ReadAttribute(file, "Cosmology", "Omega_lambda", H5T_NATIVE_DOUBLE, &Header.OmegaLambda0);  
  ReadAttribute(file, "Cosmology", "h", H5T_NATIVE_DOUBLE, &Header.h);  
  for(int i=0; i<TypeMax; i+=1)
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
  Header.length_conversion   = (length_cgs / (1.0e6*parsec_cgs)) * Header.h / HBTConfig.LengthInMpch;
  Header.mass_conversion     = (mass_cgs / solar_mass_cgs) * Header.h / HBTConfig.MassInMsunh;
  Header.velocity_conversion = (length_cgs/time_cgs) / km_cgs / HBTConfig.VelInKmS;
  Header.energy_conversion   = Header.velocity_conversion * Header.velocity_conversion;

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
  for(int i=0;i<TypeMax;i++) {
    if(i < NumPartTypes)
      Header.npartTotal[i]=(((unsigned long)NumPart_Total_HighWord[i])<<32)|NumPart_Total[i];
    else
      Header.npartTotal[i] = 0;
  }
  H5Fclose(file);
}

void SwiftSimReader_t::GetParticleCountInFile(hid_t file, int np[])
{
  int NumPartTypes;
  ReadAttribute(file, "Header", "NumPartTypes", H5T_NATIVE_INT, &NumPartTypes);
  int NumPart_ThisFile[NumPartTypes]; // same size as attributes in the file
  ReadAttribute(file, "Header", "NumPart_ThisFile", H5T_NATIVE_INT, NumPart_ThisFile);
  for(int i=0; i<TypeMax;i++)
    if(i < NumPartTypes)
      np[i] = NumPart_ThisFile[i];
    else
      np[i] = 0;
#ifdef DM_ONLY
  for(int i=0;i<TypeMax;i++)
	if(i!=TypeDM) np[i]=0;
#endif
}

HBTInt SwiftSimReader_t::CompileFileOffsets(int nfiles)
{
  HBTInt offset=0;
  np_file.reserve(nfiles);
  offset_file.reserve(nfiles);
  for(int ifile=0;ifile<nfiles;ifile++)
  {
	offset_file.push_back(offset);
	
	int np_this[TypeMax];
	hid_t file=OpenFile(ifile);
	GetParticleCountInFile(file, np_this);
	H5Fclose(file);
	HBTInt np=accumulate(begin(np_this), end(np_this), (HBTInt) 0);
	
	np_file.push_back(np);
	offset+=np;
  }
  return offset;
}

static void check_id_size(hid_t loc)
{
  hid_t dset=H5Dopen2(loc, "ParticleIDs", H5P_DEFAULT);
  hid_t dtype=H5Dget_type(dset);
  size_t ParticleIDStorageSize=H5Tget_size(dtype);
  assert(sizeof(HBTInt)>=ParticleIDStorageSize); //use HBTi8 or HBTdouble if you need long int for id
  H5Tclose(dtype);
  H5Dclose(dset);
}

void SwiftSimReader_t::ReadSnapshot(int ifile, Particle_t *ParticlesInFile, HBTInt file_start, HBTInt file_count)
{
  hid_t file=OpenFile(ifile);
  vector <int> np_this(TypeMax);
  vector <HBTInt> offset_this(TypeMax);
  GetParticleCountInFile(file, np_this.data());
  CompileOffsets(np_this, offset_this);
  
  HBTReal boxsize=Header.BoxSize;
  auto ParticlesToRead=ParticlesInFile;
  for(int itype=0;itype<TypeMax;itype++)
  {
    
    // Find the range of offsets in the file for particles of this type
    HBTInt type_first_offset = offset_this[itype];
    HBTInt type_last_offset = type_first_offset + np_this[itype] - 1;

    // Find the range of offsets in the file we actually want to read
    HBTInt read_first_offset = file_start;
    HBTInt read_last_offset = file_start + file_count - 1;

    // The overlap of these two ranges contains the particles we will read now.
    HBTInt i1 = type_first_offset;
    if(read_first_offset > i1)i1 = read_first_offset;
    HBTInt i2 = type_last_offset;
    if(read_last_offset < i2)i2 = read_last_offset;

    // Compute range of particles of this type to read from this file
    HBTInt read_offset = i1 - offset_this[itype];
    HBTInt read_count = i2 - i1 + 1;    
    if(read_count <= 0) continue;
    assert(read_offset >= 0);
    assert(read_offset + read_count <= np_this[itype]);
    
    // Open the HDF5 group for this type
    stringstream grpname;
    grpname<<"PartType"<<itype;
    hid_t particle_data=H5Gopen2(file, grpname.str().c_str(), H5P_DEFAULT);
    check_id_size(particle_data);

    const hsize_t chunksize=10*1024*1024;

    // Positions
    {
      // Check that positions are comoving
      HBTReal aexp;
      ReadAttribute(particle_data, "Coordinates", "a-scale exponent", H5T_HBTReal, &aexp);
      if(aexp!=1.0)
        {
          cout << "Can't handle Coordinates with a-scale exponent != 1\n";
          MPI_Abort(MPI_COMM_WORLD, 1);
        }

      // Read data in chunks to minimize memory overhead
      for(hsize_t offset=0; offset<read_count; offset+=chunksize)
        {
          // Read the next chunk
          hsize_t count = read_count - offset; // number left to read
          if(count > chunksize)count=chunksize;
          vector <HBTxyz> x(count);
          ReadPartialDataset(particle_data, "Coordinates", H5T_HBTReal, x.data(), offset+read_offset, count);
          // Convert to HBT units
          for(hsize_t i=0;i<count;i++)
            for(int j=0;j<3;j++)
              x[i][j] *= Header.length_conversion;
          // Box wrap if necessary
          if(HBTConfig.PeriodicBoundaryOn)
            {
              for(hsize_t i=0;i<count;i++)
                for(int j=0;j<3;j++)
                  x[i][j]=position_modulus(x[i][j], boxsize);
            }
          // Store the particle positions
          for(hsize_t i=0; i<count; i+=1)
            for(int j=0; j<3; j+=1)
              ParticlesToRead[offset+i].ComovingPosition[j] = x[i][j];
        }
    }

    // Velocities
    {
      HBTReal aexp;
      ReadAttribute(particle_data, "Velocities", "a-scale exponent", H5T_HBTReal, &aexp);

      // Read data in chunks to minimize memory overhead
      for(hsize_t offset=0; offset<read_count; offset+=chunksize)
        {
          // Read the next chunk
          hsize_t count = read_count - offset;
          if(count > chunksize)count=chunksize;
          vector <HBTxyz> v(count);
          ReadPartialDataset(particle_data, "Velocities", H5T_HBTReal, v.data(), offset+read_offset, count);
          // Convert units and store the particle velocities
          for(hsize_t i=0; i<count; i+=1)
            for(int j=0; j<3; j+=1)
              ParticlesToRead[offset+i].PhysicalVelocity[j] = v[i][j]*Header.velocity_conversion*pow(Header.ScaleFactor, aexp);
        }
    }

    // Ids
    {
      for(hsize_t offset=0; offset<read_count; offset+=chunksize)
        {
          hsize_t count = read_count - offset;
          if(count > chunksize)count=chunksize;
          vector <HBTInt> id(count);
          ReadPartialDataset(particle_data, "ParticleIDs", H5T_HBTInt, id.data(), offset+read_offset, count);
          for(hsize_t i=0; i<count; i+=1)
            ParticlesToRead[offset+i].Id=id[i];
        }
    }

    // Masses
    {
      HBTReal aexp;
      std::string name;
      if(itype==5)
        name="DynamicalMasses";
      else
        name="Masses";
      ReadAttribute(particle_data, name.c_str(), "a-scale exponent", H5T_HBTReal, &aexp);
      for(hsize_t offset=0; offset<read_count; offset+=chunksize)
        {
          hsize_t count = read_count - offset;
          if(count > chunksize)count=chunksize;
          vector <HBTReal> m(count);
          ReadPartialDataset(particle_data, name.c_str(), H5T_HBTReal, m.data(), offset+read_offset, count);
          for(hsize_t i=0; i<count; i+=1)
            ParticlesToRead[offset+i].Mass=m[i]*Header.mass_conversion*pow(Header.ScaleFactor, aexp);
        }
    }
	
#ifndef DM_ONLY
    //internal energy
#ifdef HAS_THERMAL_ENERGY
    if(itype==0)
      {
        HBTReal aexp;
        ReadAttribute(particle_data, "InternalEnergies", "a-scale exponent", H5T_HBTReal, &aexp);
        for(hsize_t offset=0; offset<read_count; offset+=chunksize)
          {
            hsize_t count = read_count - offset;
            if(count > chunksize)count=chunksize;
            vector <HBTReal> u(count);
            ReadPartialDataset(particle_data, "InternalEnergies", H5T_HBTReal, u.data(), offset+read_offset, count);
            for(hsize_t i=0; i<count; i+=1)
              ParticlesToRead[offset+i].InternalEnergy=u[i]*Header.energy_conversion*pow(Header.ScaleFactor, aexp);
          }
      }
#endif
    {//type
      ParticleType_t t=static_cast<ParticleType_t>(itype);
      for(hsize_t i=0;i<read_count;i++)
        ParticlesToRead[i].Type=t;
    }
#endif

    // Advance to next particle type
    H5Gclose(particle_data);
    ParticlesToRead += read_count;
  }
  H5Fclose(file);
}

void SwiftSimReader_t::ReadGroupParticles(int ifile, SwiftParticleHost_t *ParticlesInFile,
                                          HBTInt file_start, HBTInt file_count, bool FlagReadParticleId)
{
  hid_t file = OpenFile(ifile);
  vector <int> np_this(TypeMax);
  vector <HBTInt> offset_this(TypeMax);
  GetParticleCountInFile(file, np_this.data());
  CompileOffsets(np_this, offset_this);
  
  HBTReal boxsize=Header.BoxSize;
  auto ParticlesToRead=ParticlesInFile;
  for(int itype=0;itype<TypeMax;itype++)
    {
      // Find the range of offsets in the file for particles of this type
      HBTInt type_first_offset = offset_this[itype];
      HBTInt type_last_offset = type_first_offset + np_this[itype] - 1;

      // Find the range of offsets in the file we actually want to read
      HBTInt read_first_offset = file_start;
      HBTInt read_last_offset = file_start + file_count - 1;

      // The overlap of these two ranges contains the particles we will read now.
      HBTInt i1 = type_first_offset;
      if(read_first_offset > i1)i1 = read_first_offset;
      HBTInt i2 = type_last_offset;
      if(read_last_offset < i2)i2 = read_last_offset;

      // Compute range of particles of this type to read from this file
      HBTInt read_offset = i1 - offset_this[itype];
      HBTInt read_count = i2 - i1 + 1;    
      if(read_count <= 0) continue;
      assert(read_offset >= 0);
      assert(read_offset + read_count <= np_this[itype]);

      // Open the HDF5 group for this particle type
      stringstream grpname;
      grpname<<"PartType"<<itype;
      hid_t particle_data=H5Gopen2(file, grpname.str().c_str(), H5P_DEFAULT);
        
      const hsize_t chunksize=10*1024*1024;
	
      if(FlagReadParticleId)
        {

          // Positions
          {
            // Check that positions are comoving
            HBTReal aexp;
            ReadAttribute(particle_data, "Coordinates", "a-scale exponent", H5T_HBTReal, &aexp);
            if(aexp!=1.0)
              {
                cout << "Can't handle Coordinates with a-scale exponent != 1\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
              }
            
            // Read data in chunks to minimize memory overhead
            for(hsize_t offset=0; offset<read_count; offset+=chunksize)
              {
                // Read the next chunk
                hsize_t count = read_count - offset;
                if(count > chunksize)count=chunksize;
                vector <HBTxyz> x(count);
                ReadPartialDataset(particle_data, "Coordinates", H5T_HBTReal, x.data(), offset+read_offset, count);
                // Convert to HBT units
                for(hsize_t i=0;i<count;i++)
                  for(int j=0;j<3;j++)
                    x[i][j] *= Header.length_conversion;
                // Box wrap if necessary
                if(HBTConfig.PeriodicBoundaryOn)
                  {
                    for(hsize_t i=0;i<count;i++)
                      for(int j=0;j<3;j++)
                        x[i][j]=position_modulus(x[i][j], boxsize);
                  }
                // Store the particle positions
                for(hsize_t i=0; i<count; i+=1)
                  for(int j=0; j<3; j+=1)
                    ParticlesToRead[offset+i].ComovingPosition[j] = x[i][j];
              }
          }

          // Velocities
          {
            HBTReal aexp;
            ReadAttribute(particle_data, "Velocities", "a-scale exponent", H5T_HBTReal, &aexp);

            // Read data in chunks to minimize memory overhead
            for(hsize_t offset=0; offset<read_count; offset+=chunksize)
              {
                // Read the next chunk
                hsize_t count = read_count - offset;
                if(count > chunksize)count=chunksize;
                vector <HBTxyz> v(count);
                ReadPartialDataset(particle_data, "Velocities", H5T_HBTReal, v.data(), offset+read_offset, count);
                // Convert units and store the particle velocities
                for(hsize_t i=0; i<count; i+=1)
                  for(int j=0; j<3; j+=1)
                    ParticlesToRead[offset+i].PhysicalVelocity[j] = v[i][j]*Header.velocity_conversion*pow(Header.ScaleFactor, aexp);
              }
          }

          // Ids
          {
            for(hsize_t offset=0; offset<read_count; offset+=chunksize)
              {
                hsize_t count = read_count - offset;
                if(count > chunksize)count=chunksize;
                vector <HBTInt> id(count);
                ReadPartialDataset(particle_data, "ParticleIDs", H5T_HBTInt, id.data(), offset+read_offset, count);
                for(hsize_t i=0; i<count; i+=1)
                  ParticlesToRead[offset+i].Id=id[i];
              }
          }

          // Masses
          {
            HBTReal aexp;
            std::string name;
            if(itype==5)
              name="DynamicalMasses";
            else
              name="Masses";
            ReadAttribute(particle_data, name.c_str(), "a-scale exponent", H5T_HBTReal, &aexp);
            for(hsize_t offset=0; offset<read_count; offset+=chunksize)
              {
                hsize_t count = read_count - offset;
                if(count > chunksize)count=chunksize;
                vector <HBTReal> m(count);
                ReadPartialDataset(particle_data, name.c_str(), H5T_HBTReal, m.data(), offset+read_offset, count);
                for(hsize_t i=0; i<count; i+=1)
                  ParticlesToRead[offset+i].Mass=m[i]*Header.mass_conversion*pow(Header.ScaleFactor, aexp);
              }
          }

#ifndef DM_ONLY
          //internal energy
#ifdef HAS_THERMAL_ENERGY
          if(itype==0)
            {
              HBTReal aexp;
              ReadAttribute(particle_data, "InternalEnergies", "a-scale exponent", H5T_HBTReal, &aexp);
              for(hsize_t offset=0; offset<read_count; offset+=chunksize)
                {
                  hsize_t count = read_count - offset;
                  if(count > chunksize)count=chunksize;
                  vector <HBTReal> u(count);
                  ReadPartialDataset(particle_data, "InternalEnergies", H5T_HBTReal, u.data(), offset+read_offset, count);
                  for(hsize_t i=0; i<count; i+=1)
                    ParticlesToRead[offset+i].InternalEnergy=u[i]*Header.energy_conversion*pow(Header.ScaleFactor, aexp);
                }
            }
#endif
          {//type
            ParticleType_t t=static_cast<ParticleType_t>(itype);
            for(int i=0;i<read_count;i++)
              ParticlesToRead[i].Type=t;
          }
#endif
        }
      
      // Hostid
      {
        for(hsize_t offset=0; offset<read_count; offset+=chunksize)
          {
            hsize_t count = read_count - offset;
            if(count > chunksize)count=chunksize;
            vector <HBTInt> id(count);
            ReadPartialDataset(particle_data, "FOFGroupIDs", H5T_HBTInt, id.data(), offset+read_offset, count);
            for(hsize_t i=0; i<count; i+=1)
              ParticlesToRead[offset+i].HostId=id[i];
          }
      }
      // Advance to next particle type
      ParticlesToRead += read_count;
      H5Gclose(particle_data);
    }
  H5Fclose(file);
}

void SwiftSimReader_t::LoadSnapshot(MpiWorker_t &world, int snapshotId, vector <Particle_t> &Particles, Cosmology_t &Cosmology)
{  
  MPI_Barrier(world.Communicator);

  // Decide how many ranks per node read simultaneously
  int nr_nodes = (world.size() / world.MaxNodeSize);
  int nr_reading = HBTConfig.MaxConcurrentIO / nr_nodes;
  if(nr_reading < 1)nr_reading = 1; // Always at least one per node
  
  SetSnapshot(snapshotId);
  
  const int root=0;
  if(world.rank()==root)
  {
    ReadHeader(0, Header);
    CompileFileOffsets(Header.NumberOfFiles);

    /* Report conversion factors used to go from SWIFT to HBT units */
    cout << "Conversion factor from SWIFT length units to " << HBTConfig.LengthInMpch << " Mpc/h = " << Header.length_conversion << endl;
    cout << "Conversion factor from SWIFT mass units to " << HBTConfig.MassInMsunh << " Msun/h = " << Header.mass_conversion << endl;
    cout << "Conversion factor from SWIFT velocity units to " << HBTConfig.VelInKmS << " km/s = " << Header.velocity_conversion << endl;
    cout << "Null group ID is " << Header.NullGroupId << endl;
    cout << "Number of ranks per node reading simultaneously is " << nr_reading << endl;
  }
  MPI_Bcast(&Header, 1, MPI_SwiftSimHeader_t, root, world.Communicator);
  world.SyncContainer(np_file, MPI_HBT_INT, root);
  world.SyncContainer(offset_file, MPI_HBT_INT, root);
  
  Cosmology.Set(Header.ScaleFactor, Header.OmegaM0, Header.OmegaLambda0);

  // Decide how many particles this MPI rank will read
  HBTInt np_total = accumulate(np_file.begin(), np_file.end(), (HBTInt) 0);
  HBTInt np_local = np_total / world.size();
  if(world.rank() < (np_total % world.size()))np_local += 1;
#ifndef NDEBUG
  HBTInt np_check;
  MPI_Allreduce(&np_local, &np_check, 1, MPI_HBT_INT, MPI_SUM, world.Communicator);
  assert(np_check==np_total);
#endif
  
  // Determine offset to the first and last particle this rank will read
  HBTInt local_first_offset;
  MPI_Scan(&np_local, &local_first_offset, 1, MPI_HBT_INT, MPI_SUM, world.Communicator);
  local_first_offset -= np_local;
  HBTInt local_last_offset = local_first_offset + np_local - 1;
  assert(local_first_offset>=0);
  assert(local_last_offset<np_total);
  
  // Allocate storage for the particles
  Particles.resize(np_local);

  // Allow a limited number of ranks per node to read simultaneously
  int reads_done = 0;
  for(int rank_within_node=0; rank_within_node < world.MaxNodeSize; rank_within_node+=1) {
    if(rank_within_node == world.NodeRank) {
      
      // Loop over all files
      HBTInt particle_offset = 0;
      for(int file_nr=0; file_nr<Header.NumberOfFiles; file_nr+=1) {

        // Determine global offset of first particle to read from this file:
        // This is the larger of the offset of the first particle in the file
        // and the offset of the first particle this rank is to read.
        HBTInt i1 = offset_file[file_nr];
        if(local_first_offset > i1)i1 = local_first_offset;
    
        // Determine global offset of last particle to read from this file:
        // This is the smaller of the offset to the last particle in this file
        // and the offset of the last particle this rank is to read.
        HBTInt i2 = offset_file[file_nr] + np_file[file_nr] - 1;
        if(local_last_offset < i2)i2 = local_last_offset;
    
        if(i2 >= i1) {
          // We have particles to read from this file.
          HBTInt file_start = i1 - offset_file[file_nr]; // Offset to first particle to read
          HBTInt file_count = i2 - i1 + 1;               // Number of particles to read
          assert(file_count > 0);
          assert(file_start >= 0);
          assert(file_start+file_count <= np_file[file_nr]);
          ReadSnapshot(file_nr, Particles.data()+particle_offset, file_start, file_count);
          particle_offset += file_count;
        } 
      } // Next file
      assert(particle_offset==np_local); // Check we read the expected number of particles
      reads_done += 1;
    }
    if(rank_within_node % nr_reading == nr_reading-1)MPI_Barrier(world.Communicator);
  } // Next MPI rank within the node

  // Every rank should have executed the reading code exactly once
  assert(reads_done==1);
  
  //#define SNAPSHOT_IO_TEST
#ifdef SNAPSHOT_IO_TEST
  // For testing: dump the snapshot to a new set of files
  // Generate test file name for this MPI  rank
  stringstream formatter1;
  formatter1<<HBTConfig.SubhaloPath<<"/"<<setw(3)<<setfill('0')<<snapshotId<<"/"<<"test_"<<setw(3)<<setfill('0')<<snapshotId<<"."<<world.rank()<<".hdf5";
  string tfilename = formatter1.str();
  // Create array of coordinates
  double *pos = (double *) malloc(3*sizeof(double)*np_local);
  for(size_t i = 0; i<np_local; i+=1) {
    pos[3*i+0] = Particles[i].ComovingPosition[0];
    pos[3*i+1] = Particles[i].ComovingPosition[1];
    pos[3*i+2] = Particles[i].ComovingPosition[2];
  }
  // Create array of IDs
  long long *ids = (long long *) malloc(sizeof(long long)*np_local);
  for(size_t i = 0; i<np_local; i+=1)
    ids[i] = Particles[i].Id;
  // Create array of types
  int *type = (int *) malloc(sizeof(int)*np_local);
  for(size_t i = 0; i<np_local; i+=1)
    type[i] = Particles[i].Type;

  // Create the file
  hid_t tfile = H5Fcreate(tfilename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  // Write out the data
  hsize_t ndims;
  hsize_t dims[2];
  ndims = 2;
  dims[0] = np_local;
  dims[1] = 3;
  writeHDFmatrix(tfile, pos, "Coordinates", ndims, dims, H5T_NATIVE_DOUBLE);
  ndims = 1;
  writeHDFmatrix(tfile, ids, "ParticleIDs", ndims, dims, H5T_NATIVE_LLONG);
  writeHDFmatrix(tfile, type, "Types", ndims, dims, H5T_NATIVE_INT);
  
  // Tidy up
  H5Fclose(tfile);
  free(pos);
  free(ids);
  free(type);
  
#endif
  
}

inline bool CompParticleHost(const SwiftParticleHost_t &a, const SwiftParticleHost_t &b)
{
  return a.HostId<b.HostId;
}

void SwiftSimReader_t::LoadGroups(MpiWorker_t &world, int snapshotId, vector< Halo_t >& Halos)
{//read in particle properties at the same time, to avoid particle look-up at later stage.

  SetSnapshot(snapshotId);

  // Decide how many ranks per node read simultaneously
  int nr_nodes = (world.size() / world.MaxNodeSize);
  int nr_reading = HBTConfig.MaxConcurrentIO / nr_nodes;
  if(nr_reading < 1)nr_reading = 1; // Always at least one per node
  
  const int root=0;
  if(world.rank()==root)
  {
    ReadHeader(0, Header);
    CompileFileOffsets(Header.NumberOfFiles);
  }
  MPI_Bcast(&Header, 1, MPI_SwiftSimHeader_t, root, world.Communicator);
  world.SyncContainer(np_file, MPI_HBT_INT, root);
  world.SyncContainer(offset_file, MPI_HBT_INT, root);

  // Decide how many particles this MPI rank will read
  HBTInt np_total = accumulate(np_file.begin(), np_file.end(), (HBTInt) 0);
  HBTInt np_local = np_total / world.size();
  if(world.rank() < (np_total % world.size()))np_local += 1;
#ifndef NDEBUG
  HBTInt np_check;
  MPI_Allreduce(&np_local, &np_check, 1, MPI_HBT_INT, MPI_SUM, world.Communicator);
  assert(np_check==np_total);
#endif
  
  // Determine offset to the first and last particle this rank will read
  HBTInt local_first_offset;
  MPI_Scan(&np_local, &local_first_offset, 1, MPI_HBT_INT, MPI_SUM, world.Communicator);
  local_first_offset -= np_local;
  HBTInt local_last_offset = local_first_offset + np_local - 1;
  assert(local_first_offset>=0);
  assert(local_last_offset<np_total);
  
  // Allocate storage for the particles
  vector <SwiftParticleHost_t> ParticleHosts;
  ParticleHosts.resize(np_local);

  bool FlagReadId=true; //!HBTConfig.GroupLoadedIndex;

  // Allow a limited number of ranks per node to read simultaneously
  int reads_done = 0;
  for(int rank_within_node=0; rank_within_node < world.MaxNodeSize; rank_within_node+=1) {
    if(rank_within_node == world.NodeRank) {

      // Loop over all files
      HBTInt particle_offset = 0;
      for(int file_nr=0; file_nr<Header.NumberOfFiles; file_nr+=1) {

        // Determine global offset of first particle to read from this file:
        // This is the larger of the offset of the first particle in the file
        // and the offset of the first particle this rank is to read.
        HBTInt i1 = offset_file[file_nr];
        if(local_first_offset > i1)i1 = local_first_offset;
    
        // Determine global offset of last particle to read from this file:
        // This is the smaller of the offset to the last particle in this file
        // and the offset of the last particle this rank is to read.
        HBTInt i2 = offset_file[file_nr] + np_file[file_nr] - 1;
        if(local_last_offset < i2)i2 = local_last_offset;
    
        if(i2 >= i1) {
          // We have particles to read from this file.
          HBTInt file_start = i1 - offset_file[file_nr]; // Offset to first particle to read
          HBTInt file_count = i2 - i1 + 1;               // Number of particles to read
          assert(file_count > 0);
          assert(file_start >= 0);
          assert(file_start+file_count <= np_file[file_nr]);
          ReadGroupParticles(file_nr, ParticleHosts.data()+particle_offset, file_start, file_count, FlagReadId);
          particle_offset += file_count;
        } 
      } // Next file
      assert(particle_offset==np_local); // Check we read the expected number of particles
      reads_done += 1;
    }
    if(rank_within_node % nr_reading == nr_reading-1)MPI_Barrier(world.Communicator);
  } // Next MPI rank within the node

  // Every rank should have executed the reading code exactly once
  assert(reads_done==1);

  //#define HALO_IO_TEST
#ifdef HALO_IO_TEST
  //
  // For testing: dump the snapshot to a new set of files
  //
  // Generate test file name for this MPI  rank
  stringstream formatter1;
  formatter1<<HBTConfig.SubhaloPath<<"/"<<setw(3)<<setfill('0')<<snapshotId<<"/"<<"test_halo_"<<setw(3)<<setfill('0')<<snapshotId<<"."<<world.rank()<<".hdf5";
  string tfilename = formatter1.str();
  // Create array of coordinates
  double *pos = (double *) malloc(3*sizeof(double)*np_local);
  for(size_t i = 0; i<np_local; i+=1) {
    pos[3*i+0] = ParticleHosts[i].ComovingPosition[0];
    pos[3*i+1] = ParticleHosts[i].ComovingPosition[1];
    pos[3*i+2] = ParticleHosts[i].ComovingPosition[2];
  }
  // Create array of IDs
  long long *ids = (long long *) malloc(sizeof(long long)*np_local);
  for(size_t i = 0; i<np_local; i+=1)
    ids[i] = ParticleHosts[i].Id;
  // Create array of types
  int *type = (int *) malloc(sizeof(int)*np_local);
  for(size_t i = 0; i<np_local; i+=1)
    type[i] = ParticleHosts[i].Type;
  // Create array of group indexes
  int *fofnr = (int *) malloc(sizeof(int)*np_local);
  for(size_t i = 0; i<np_local; i+=1)
    fofnr[i] = ParticleHosts[i].HostId;
  
  // Create the file
  hid_t tfile = H5Fcreate(tfilename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  // Write out the data
  hsize_t ndims;
  hsize_t dims[2];
  ndims = 2;
  dims[0] = np_local;
  dims[1] = 3;
  writeHDFmatrix(tfile, pos, "Coordinates", ndims, dims, H5T_NATIVE_DOUBLE);
  ndims = 1;
  writeHDFmatrix(tfile, ids, "ParticleIDs", ndims, dims, H5T_NATIVE_LLONG);
  writeHDFmatrix(tfile, type, "Types", ndims, dims, H5T_NATIVE_INT);
  writeHDFmatrix(tfile, fofnr, "FoFNr", ndims, dims, H5T_NATIVE_INT);
  
  // Tidy up
  H5Fclose(tfile);
  free(pos);
  free(ids);
  free(type);
  free(fofnr);
  //
  // END OF TEST CODE
  //
#endif
  
  // Sort particles by host
  sort(ParticleHosts.begin(), ParticleHosts.end(), CompParticleHost);
  if(!ParticleHosts.empty())
  {
    assert(ParticleHosts.back().HostId<=Header.NullGroupId);//max haloid==NullGroupId
    assert(ParticleHosts.front().HostId>=0);//min haloid>=0
  }
  
  struct HaloLen_t
  {
    HBTInt haloid;
    HBTInt np;
    HaloLen_t(){};
    HaloLen_t(HBTInt haloid, HBTInt np): haloid(haloid), np(np)
    {
    }
  };
  vector <HaloLen_t> HaloLen;
  
  HBTInt curr_host_id=Header.NullGroupId;
  for(auto &&p: ParticleHosts)
  {
    if(p.HostId==Header.NullGroupId) break;//NullGroupId comes last
    if(p.HostId!=curr_host_id)
    {
      curr_host_id=p.HostId;
      HaloLen.emplace_back(curr_host_id, 1);
    }
    else
      HaloLen.back().np++;
  }
  Halos.resize(HaloLen.size());
  for(HBTInt i=0;i<Halos.size();i++)
  {
    Halos[i].HaloId=HaloLen[i].haloid;
    Halos[i].Particles.resize(HaloLen[i].np);
  }
  auto p_in=ParticleHosts.begin();
  for(auto &&h: Halos)
  {
    for(auto &&p: h.Particles)
    {
      p=*p_in;
      ++p_in;
    }
  }
  
  VectorFree(ParticleHosts);
  
  ExchangeAndMerge(world, Halos);
  
  HBTConfig.GroupLoadedFullParticle = true;

}

bool IsSwiftSimGroup(const string &GroupFileFormat)
{
  return GroupFileFormat.substr(0, 8)=="swiftsim";
}
