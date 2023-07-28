using namespace std;
#include <iostream>
#include <numeric>
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

void create_SwiftSimHeader_MPI_type(MPI_Datatype& dtype)
{
  /*to create the struct data type for communication*/	
  SwiftSimHeader_t p;
  #define NumAttr 13
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
  RegisterAttr(mass, MPI_DOUBLE, TypeMax)
  RegisterAttr(npart[0], MPI_INT, TypeMax)
  RegisterAttr(npartTotal[0], MPI_HBT_INT, TypeMax)
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
  if(Header.BoxSize!=HBTConfig.BoxSize) {
    cout << "Box size in snapshot does not match parameter file!\n";
    MPI_Abort(MPI_COMM_WORLD, 1);  
  }
  ReadAttribute(file, "Cosmology", "Scale-factor", H5T_NATIVE_DOUBLE, &Header.ScaleFactor);
  ReadAttribute(file, "Cosmology", "Omega_m", H5T_NATIVE_DOUBLE, &Header.OmegaM0);
  ReadAttribute(file, "Cosmology", "Omega_lambda", H5T_NATIVE_DOUBLE, &Header.OmegaLambda0);  
  for(int i=0; i<TypeMax; i+=1)
    Header.mass[i] = 0.0; // Swift particles always have individual masses

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
  for(int i=0;i<TypeMax;i++)
    if(i < NumPartTypes)
      Header.npartTotal[i]=(((unsigned long)NumPart_Total_HighWord[i])<<32)|NumPart_Total[i];
    else
      Header.npartTotal[i] = 0;
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
	HBTInt np=accumulate(begin(np_this), end(np_this), (HBTInt)0);
	
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

void SwiftSimReader_t::ReadSnapshot(int ifile, Particle_t *ParticlesInFile)
{
  hid_t file=OpenFile(ifile);
  vector <int> np_this(TypeMax);
  vector <HBTInt> offset_this(TypeMax);
  GetParticleCountInFile(file, np_this.data());
  CompileOffsets(np_this, offset_this);
 
  HBTReal boxsize=Header.BoxSize;
  for(int itype=0;itype<TypeMax;itype++)
  {
	int np=np_this[itype];
	if(np==0) continue;
	auto ParticlesThisType=ParticlesInFile+offset_this[itype];
	stringstream grpname;
	grpname<<"PartType"<<itype;
        const std::string& tmp = grpname.str();   
        const char* group_name = tmp.c_str();
	if(!H5Lexists(file, group_name, H5P_DEFAULT)) continue;

	hid_t particle_data=H5Gopen2(file, grpname.str().c_str(), H5P_DEFAULT);
// 	if(particle_data<0) continue; //skip non-existing type

	check_id_size(particle_data);

        // Positions
        {
          vector <HBTxyz> x(np);
          ReadDataset(particle_data, "Coordinates", H5T_HBTReal, x.data());
          HBTReal aexp;
          ReadAttribute(particle_data, "Coordinates", "a-scale exponent", H5T_HBTReal, &aexp);
          if(aexp!=1.0)
            {
              /* Don't know how to do the box wrapping in this case! Is BoxSize comoving? */
              cout << "Can't handle Coordinates with a-scale exponent != 1\n";
              MPI_Abort(MPI_COMM_WORLD, 1);
            }
          if(HBTConfig.PeriodicBoundaryOn)
            {
              for(int i=0;i<np;i++)
                for(int j=0;j<3;j++)
                  x[i][j]=position_modulus(x[i][j], boxsize);
            }
          for(int i=0;i<np;i++)
            for(int j=0; j<3; j+=1)
              ParticlesThisType[i].ComovingPosition[j] = x[i][j] * pow(Header.ScaleFactor, aexp-1.0);
        }

        // Velocities
        {
          vector <HBTxyz> v(np);
          ReadDataset(particle_data, "Velocities", H5T_HBTReal, v.data());
          HBTReal aexp;
          ReadAttribute(particle_data, "Velocities", "a-scale exponent", H5T_HBTReal, &aexp);
          for(int i=0;i<np;i++)
            for(int j=0;j<3;j++)
              ParticlesThisType[i].PhysicalVelocity[j]=v[i][j]*pow(Header.ScaleFactor, aexp);
        }

        // Ids
        {
          vector <HBTInt> id(np);
          ReadDataset(particle_data, "ParticleIDs", H5T_HBTInt, id.data());
          for(int i=0;i<np;i++)
            ParticlesThisType[i].Id=id[i];
        }

        // Masses
        {
          vector <HBTReal> m(np);
          ReadDataset(particle_data, "Masses", H5T_HBTReal, m.data());
          HBTReal aexp;
          ReadAttribute(particle_data, "Masses", "a-scale exponent", H5T_HBTReal, &aexp);
          for(int i=0;i<np;i++)
            ParticlesThisType[i].Mass=m[i]*pow(Header.ScaleFactor, aexp);
        }
	
#ifndef DM_ONLY
	//internal energy
#ifdef HAS_THERMAL_ENERGY
	if(itype==0)
          {
            // Here we convert the internal energy to physical units.
            // TODO: is this a dependence correct? Should I be ensuring it's in velocity units squared?
            // Note that swift has 'a-scale exponent'=-2 for this dataset.
            HBTReal aexp;
            ReadAttribute(particle_data, "InternalEnergies", "a-scale exponent", H5T_HBTReal, &aexp);
            vector <HBTReal> u(np);
            ReadDataset(particle_data, "InternalEnergies", H5T_HBTReal, u.data());
            for(int i=0;i<np;i++)
              ParticlesThisType[i].InternalEnergy=u[i]*pow(Header.ScaleFactor, aexp);
          }
#endif
	{//type
	  ParticleType_t t=static_cast<ParticleType_t>(itype);
	  for(int i=0;i<np;i++)
		ParticlesThisType[i].Type=t;
	}
#endif
	H5Gclose(particle_data);
  }
  H5Fclose(file);
}

void SwiftSimReader_t::ReadGroupParticles(int ifile, SwiftParticleHost_t *ParticlesInFile, bool FlagReadParticleId)
{
  hid_t file = OpenFile(ifile);
  vector <int> np_this(TypeMax);
  vector <HBTInt> offset_this(TypeMax);
  GetParticleCountInFile(file, np_this.data());
  CompileOffsets(np_this, offset_this);
  
  HBTReal boxsize=Header.BoxSize;
  for(int itype=0;itype<TypeMax;itype++)
  {
	int np=np_this[itype];
	if(np==0) continue;
	auto ParticlesThisType=ParticlesInFile+offset_this[itype];
	stringstream grpname;
	grpname<<"PartType"<<itype;
	if(!H5Lexists(file, grpname.str().c_str(), H5P_DEFAULT)) continue;
	hid_t particle_data=H5Gopen2(file, grpname.str().c_str(), H5P_DEFAULT);
	
	if(FlagReadParticleId)
	{
          // Positions
          {
            vector <HBTxyz> x(np);
            ReadDataset(particle_data, "Coordinates", H5T_HBTReal, x.data());
            HBTReal aexp;
            ReadAttribute(particle_data, "Coordinates", "a-scale exponent", H5T_HBTReal, &aexp);
            if(aexp!=1.0)
              {
                /* Don't know how to do the box wrapping in this case! Is BoxSize comoving? */
                cout << "Can't handle Coordinates with a-scale exponent != 1\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
              }
            if(HBTConfig.PeriodicBoundaryOn)
              {
                for(int i=0;i<np;i++)
                  for(int j=0;j<3;j++)
                    x[i][j]=position_modulus(x[i][j], boxsize);
              }
            for(int i=0;i<np;i++)
              for(int j=0; j<3; j+=1)
                ParticlesThisType[i].ComovingPosition[j] = x[i][j] * pow(Header.ScaleFactor, aexp-1.0);
          }
          
          // Velocities
          {
            vector <HBTxyz> v(np);
            ReadDataset(particle_data, "Velocities", H5T_HBTReal, v.data());
            HBTReal aexp;
            ReadAttribute(particle_data, "Velocities", "a-scale exponent", H5T_HBTReal, &aexp);
            for(int i=0;i<np;i++)
              for(int j=0;j<3;j++)
                ParticlesThisType[i].PhysicalVelocity[j]=v[i][j]*pow(Header.ScaleFactor, aexp);
          }

          // Ids
          {
            vector <HBTInt> id(np);
            ReadDataset(particle_data, "ParticleIDs", H5T_HBTInt, id.data());
            for(int i=0;i<np;i++)
              ParticlesThisType[i].Id=id[i];
          }

          // Masses
          {
            vector <HBTReal> m(np);
            HBTReal aexp;
            if(itype == 5) {
              /* Use dynamical mass field for black hole particles */
              ReadDataset(particle_data, "DynamicalMasses", H5T_HBTReal, m.data());
              ReadAttribute(particle_data, "DynamicalMasses", "a-scale exponent", H5T_HBTReal, &aexp);
            } else {
              ReadDataset(particle_data, "Masses", H5T_HBTReal, m.data());
              ReadAttribute(particle_data, "Masses", "a-scale exponent", H5T_HBTReal, &aexp);              
            }
            for(int i=0;i<np;i++)
              ParticlesThisType[i].Mass=m[i]*pow(Header.ScaleFactor, aexp);
          }

#ifndef DM_ONLY
          //internal energy
#ifdef HAS_THERMAL_ENERGY
          if(itype==0)
          {
            // Here we convert the internal energy to physical units.
            // TODO: is this a dependence correct? Should I be ensuring it's in velocity units squared?
            // Note that swift has 'a-scale exponent'=-2 for this dataset.
            HBTReal aexp;
            ReadAttribute(particle_data, "InternalEnergies", "a-scale exponent", H5T_HBTReal, &aexp);
            vector <HBTReal> u(np);
            ReadDataset(particle_data, "InternalEnergies", H5T_HBTReal, u.data());
            for(int i=0;i<np;i++)
              ParticlesThisType[i].InternalEnergy=u[i]*pow(Header.ScaleFactor, aexp);
          }
#endif
          {//type
            ParticleType_t t=static_cast<ParticleType_t>(itype);
            for(int i=0;i<np;i++)
              ParticlesThisType[i].Type=t;
          }
#endif
	}
	
	{//Hostid
	  vector <HBTInt> id(np);
	  ReadDataset(particle_data, "FOFGroupIDs", H5T_HBTInt, id.data());
	  for(int i=0;i<np;i++)
		ParticlesThisType[i].HostId=(id[i]<0?NullGroupId:id[i]);//negative means outside fof but within Rv 
	}
	
	H5Gclose(particle_data);
  }
  
  H5Fclose(file);
}


void SwiftSimReader_t::ReadUnits(HBTReal &MassInMsunh, HBTReal &LengthInMpch, HBTReal &VelInKmS)
{
  // Read Swift unit information to determine how HBT's unit parameters should be set
  hid_t file = OpenFile(0);
  double h;
  // Hubble parameter
  ReadAttribute(file, "Cosmology", "h", H5T_NATIVE_DOUBLE, &h);
  // Constants we'll need
  double parsec_cgs, solar_mass_cgs;
  ReadAttribute(file, "PhysicalConstants/CGS", "parsec", H5T_NATIVE_DOUBLE,
                &parsec_cgs);
  ReadAttribute(file, "PhysicalConstants/CGS", "solar_mass", H5T_NATIVE_DOUBLE,
                &solar_mass_cgs);
  // Coordinates
  double length_cgs, length_hexp;
  ReadAttribute(file, "PartType1/Coordinates",
                "Conversion factor to CGS (not including cosmological corrections)",
                H5T_NATIVE_DOUBLE, &length_cgs);
  ReadAttribute(file, "PartType1/Coordinates", "h-scale exponent", H5T_NATIVE_DOUBLE,
                &length_hexp);
  // Velocities
  double velocity_cgs, velocity_hexp;
  ReadAttribute(file, "PartType1/Velocities",
                "Conversion factor to CGS (not including cosmological corrections)",
                H5T_NATIVE_DOUBLE, &velocity_cgs);
  ReadAttribute(file, "PartType1/Velocities", "h-scale exponent", H5T_NATIVE_DOUBLE,
                &velocity_hexp);
  // Masses
  double mass_cgs, mass_hexp;
  ReadAttribute(file, "PartType1/Masses",
                "Conversion factor to CGS (not including cosmological corrections)",
                H5T_NATIVE_DOUBLE, &mass_cgs);
  ReadAttribute(file, "PartType1/Masses", "h-scale exponent", H5T_NATIVE_DOUBLE,
                &mass_hexp);

  // Conversion factor from mass units in snapshot to to Msolar/h
  MassInMsunh = (mass_cgs/solar_mass_cgs)*pow(h, mass_hexp)*h;

  // Conversion factor from length units in snapshot to Mpc/h
  LengthInMpch = (length_cgs/(1.0e6*parsec_cgs))*pow(h, length_hexp)*h;

  // Conversion factor from velocity units in snapshot to km/sec
  VelInKmS = (velocity_cgs/1.0e5)*pow(h, velocity_hexp);
  
  H5Fclose(file);
}


void SwiftSimReader_t::LoadSnapshot(MpiWorker_t &world, int snapshotId, vector <Particle_t> &Particles, Cosmology_t &Cosmology)
{

  MPI_Barrier(world.Communicator);

  SetSnapshot(snapshotId);
  
  const int root=0;
  if(world.rank()==root)
  {
    ReadHeader(0, Header);
    CompileFileOffsets(Header.NumberOfFiles);

    // Report units
    HBTReal MassInMsunh;
    HBTReal LengthInMpch;
    HBTReal VelInKmS;
    ReadUnits(MassInMsunh, LengthInMpch, VelInKmS);
    cout << "MassInMsunh  = " << MassInMsunh  << endl;
    cout << "LengthInMpch = " << LengthInMpch << endl;
    cout << "VelInKmS     = " << VelInKmS     << endl;
  }
  MPI_Bcast(&Header, 1, MPI_SwiftSimHeader_t, root, world.Communicator);
  world.SyncContainer(np_file, MPI_HBT_INT, root);
  world.SyncContainer(offset_file, MPI_HBT_INT, root);
  
  Cosmology.Set(Header.ScaleFactor, Header.OmegaM0, Header.OmegaLambda0);
  
  HBTInt nfiles_skip, nfiles_end;
  AssignTasks(world.rank(), world.size(), Header.NumberOfFiles, nfiles_skip, nfiles_end);
  {
    HBTInt np=0;
    np=accumulate(np_file.begin()+nfiles_skip, np_file.begin()+nfiles_end, np);
    Particles.resize(np);
  }
  
  for(int i=0, ireader=0;i<world.size();i++, ireader++)
  {
	if(ireader==HBTConfig.MaxConcurrentIO) 
	{
	  ireader=0;//reset reader count
	  MPI_Barrier(world.Communicator);//wait for every thread to arrive.
	}
	if(i==world.rank())//read
	{
	  for(int iFile=nfiles_skip; iFile<nfiles_end; iFile++)
	  {
		ReadSnapshot(iFile, Particles.data()+offset_file[iFile]-offset_file[nfiles_skip]);
	  }
	}
  }

  MPI_Barrier(world.Communicator);

}

inline bool CompParticleHost(const SwiftParticleHost_t &a, const SwiftParticleHost_t &b)
{
  return a.HostId<b.HostId;
}

void SwiftSimReader_t::LoadGroups(MpiWorker_t &world, int snapshotId, vector< Halo_t >& Halos)
{//read in particle properties at the same time, to avoid particle look-up at later stage.
  SetSnapshot(snapshotId);
  
  const int root=0;
  if(world.rank()==root)
  {
    ReadHeader(0, Header);
    CompileFileOffsets(Header.NumberOfFiles);
  }
  MPI_Bcast(&Header, 1, MPI_SwiftSimHeader_t, root, world.Communicator);
  world.SyncContainer(np_file, MPI_HBT_INT, root);
  world.SyncContainer(offset_file, MPI_HBT_INT, root);
  
  vector <SwiftParticleHost_t> ParticleHosts;
  HBTInt nfiles_skip, nfiles_end;
  AssignTasks(world.rank(), world.size(), Header.NumberOfFiles, nfiles_skip, nfiles_end);
  {
    HBTInt np=0;
    np=accumulate(np_file.begin()+nfiles_skip, np_file.begin()+nfiles_end, np);
    ParticleHosts.resize(np);
  }
  bool FlagReadId=true; //!HBTConfig.GroupLoadedIndex;
  
  for(int i=0, ireader=0;i<world.size();i++, ireader++)
  {
	if(ireader==HBTConfig.MaxConcurrentIO) 
	{
	  ireader=0;//reset reader count
	  MPI_Barrier(world.Communicator);//wait for every thread to arrive.
	}
	if(i==world.rank())//read
	{
	  for(int iFile=nfiles_skip; iFile<nfiles_end; iFile++)
		ReadGroupParticles(iFile, ParticleHosts.data()+offset_file[iFile]-offset_file[nfiles_skip], FlagReadId);
	}
  }
  
  sort(ParticleHosts.begin(), ParticleHosts.end(), CompParticleHost);
  if(!ParticleHosts.empty())
  {
    assert(ParticleHosts.back().HostId<=NullGroupId);//max haloid==NullGroupId
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
  
  HBTInt curr_host_id=NullGroupId;
  for(auto &&p: ParticleHosts)
  {
    if(p.HostId==NullGroupId) break;//NullGroupId comes last
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
  return GroupFileFormat.substr(0, 8)=="swiftsim";
}
