/* IO for SwiftSim data.
 *
 * To specify a list of snapshot, list the snapshot directories (one per line) in snapshotlist.txt and place it under
 * your subhalo output directory.
 *
 * To use this IO, in the config file, set SnapshotFormat to swiftsim,  and set GroupFileFormat to swiftsim or
 * swiftsim_particle_index.
 *
 * The groups loaded are already filled with particle properties, and the halos are distributed to processors according
 * to the CoM of each halo.
 */

#ifndef SWIFTSIM_IO_INCLUDED
#define SWIFTSIM_IO_INCLUDED
#include "../halo.h"
#include "../hdf_wrapper.h"
#include "../mpi_wrapper.h"

struct SwiftSimHeader_t
{
  int NumberOfFiles;
  double BoxSize;
  double ScaleFactor;
  double OmegaM0;
  double OmegaLambda0;
  double h;
  double mass[TypeMax];
  int npart[TypeMax];
  HBTInt npartTotal[TypeMax];
  double MassInMsunh;
  double LengthInMpch;
  double VelInKmS;
  HBTInt NullGroupId;
  double DM_comoving_softening;
  double DM_maximum_physical_softening;
  double baryon_comoving_softening;         // NOTE: currently being loaded but unused
  double baryon_maximum_physical_softening; // NOTE: currently being loaded but unused
};

void create_SwiftSimHeader_MPI_type(MPI_Datatype &dtype);

class SwiftSimReader_t
{
  string SnapshotName;
  vector<HBTInt> np_file;
  vector<HBTInt> offset_file;
  SwiftSimHeader_t Header;
  hid_t OpenFile(int ifile);
  void ReadHeader(int ifile, SwiftSimHeader_t &header);
  void ReadUnits(HBTReal &MassInMsunh, HBTReal &LengthInMpch, HBTReal &VelInKmS);
  HBTInt CompileFileOffsets(int nfiles);
  void ReadSnapshot(int ifile, Particle_t *ParticlesInFile, HBTInt file_start, HBTInt file_count);
  void ReadGroupParticles(int ifile, Particle_t *ParticlesInFile, HBTInt file_start, HBTInt file_count,
                          bool FlagReadParticleId);
  void GetFileName(int ifile, string &filename);
  void SetSnapshot(int snapshotId);
  void GetParticleCountInFile(hid_t file, int np[]);

  /* To load information about particle splits */
  void GetParticleSplitFileName(int snapshotId, string &filename);
  hid_t OpenParticleSplitFile(int snapshotId);

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
  void LoadSnapshot(MpiWorker_t &world, int snapshotId, vector<Particle_t> &Particles, Cosmology_t &Cosmology);
  void LoadGroups(MpiWorker_t &world, int snapshotId, vector<Halo_t> &Halos);
};

extern bool IsSwiftSimGroup(const string &GroupFileFormat);
#endif
