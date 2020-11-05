/* IO for SwiftSim data.
 * 
 * To specify a list of snapshot, list the snapshot directories (one per line) in snapshotlist.txt and place it under your subhalo output directory. 
 * 
 * To use this IO, in the config file, set SnapshotFormat to swiftsim,  and set GroupFileFormat to swiftsim or swiftsim_particle_index.
 * 
 * The groups loaded are already filled with particle properties, and the halos are distributed to processors according to the CoM of each halo.
 */

#ifndef SWIFTSIM_IO_INCLUDED
#define SWIFTSIM_IO_INCLUDED
#include "../hdf_wrapper.h"
#include "../halo.h"
#include "../mpi_wrapper.h"

struct SwiftSimHeader_t
{
  int      NumberOfFiles;
  double   BoxSize;
  double   ScaleFactor;
  double   OmegaM0;
  double   OmegaLambda0;
  double   mass[TypeMax];
  int      npart[TypeMax];  
  HBTInt npartTotal[TypeMax]; 
};

void create_SwiftSimHeader_MPI_type(MPI_Datatype &dtype);

struct SwiftParticleHost_t: public Particle_t
{
  HBTInt HostId;
};

class SwiftSimReader_t
{
  // TODO: read null group ID from snapshot in case user changed it in Swift config
  const int NullGroupId=(((long long) 1)<<31)-1;
  string SnapshotName;
    
  vector <HBTInt> np_file;
  vector <HBTInt> offset_file;
  SwiftSimHeader_t Header;
  hid_t OpenFile(int ifile);
  void ReadHeader(int ifile, SwiftSimHeader_t &header);
  void ReadUnits(HBTReal &MassInMsunh, HBTReal &LengthInMpch, HBTReal &VelInKmS);
  HBTInt CompileFileOffsets(int nfiles);
  void ReadSnapshot(int ifile, Particle_t * ParticlesInFile);
  void ReadGroupParticles(int ifile, SwiftParticleHost_t * ParticlesInFile, bool FlagReadParticleId);
  void GetFileName(int ifile, string &filename);
  void SetSnapshot(int snapshotId);
  void GetParticleCountInFile(hid_t file, int np[]);

  MPI_Datatype MPI_SwiftSimHeader_t;
  
public:
  SwiftSimReader_t()
  {
    create_SwiftSimHeader_MPI_type(MPI_SwiftSimHeader_t);
  }
  ~SwiftSimReader_t()
  {
    MPI_Type_free(&MPI_SwiftSimHeader_t);
  }
  void LoadSnapshot(MpiWorker_t &world, int snapshotId, vector <Particle_t> &Particles, Cosmology_t &Cosmology);
  void LoadGroups(MpiWorker_t &world, int snapshotId, vector <Halo_t> &Halos);
};

extern bool IsSwiftSimGroup(const string &GroupFileFormat);
#endif
