#include "config_parser.h"
#include <cstdlib>

namespace PhysicalConst
{
HBTReal G;
HBTReal H0;
} // namespace PhysicalConst

Parameter_t HBTConfig;

bool Parameter_t::TryCompulsoryParameterValue(string ParameterName, stringstream &ParameterValue)
{
#define TrySetPar(var, i)                                                                                              \
  if (ParameterName == #var)                                                                                           \
  {                                                                                                                    \
    ParameterValue >> var;                                                                                             \
    IsSet[i] = true;                                                                                                   \
    return true; /* Signals to not continue looking for matching parameter names */                                    \
  }

  TrySetPar(SnapshotPath, 0);
  TrySetPar(HaloPath, 1);
  TrySetPar(SubhaloPath, 2);
  TrySetPar(SnapshotFileBase, 3);
  TrySetPar(MaxSnapshotIndex, 4);
  TrySetPar(BoxSize, 5);
  TrySetPar(SofteningHalo, 6);

#undef TrySetPar

  return false; // This signals to continue looking for valid parameter names
}

bool Parameter_t::TrySingleValueParameter(string ParameterName, stringstream &ParameterValue)
{
#define TrySetPar(var)                                                                                                 \
  if (ParameterName == #var)                                                                                           \
  {                                                                                                                    \
    ParameterValue >> var;                                                                                             \
    return true; /* Signals to not continue looking for matching parameter names */                                    \
  }

  TrySetPar(SnapshotDirBase);
  TrySetPar(SnapshotFormat);
  TrySetPar(GroupFileFormat);
  TrySetPar(MaxConcurrentIO);
  TrySetPar(MinSnapshotIndex);
  TrySetPar(MinNumPartOfSub);
  TrySetPar(ParticleIdRankStyle);
  TrySetPar(ParticleIdNeedHash);
  TrySetPar(SnapshotIdUnsigned);
  TrySetPar(SaveSubParticleProperties);
  TrySetPar(MergeTrappedSubhalos);
  TrySetPar(MajorProgenitorMassRatio);
  TrySetPar(BoundMassPrecision);
  TrySetPar(SourceSubRelaxFactor);
  TrySetPar(SubCoreSizeFactor);
  TrySetPar(SubCoreSizeMin);
  TrySetPar(TreeAllocFactor);
  TrySetPar(TreeNodeOpenAngle);
  TrySetPar(TreeMinNumOfCells);
  TrySetPar(MaxSampleSizeOfPotentialEstimate);
  TrySetPar(RefineMostboundParticle);
  TrySetPar(MassInMsunh);
  TrySetPar(LengthInMpch);
  TrySetPar(VelInKmS);
  TrySetPar(PeriodicBoundaryOn);
  TrySetPar(SnapshotHasIdBlock);
  TrySetPar(MaxPhysicalSofteningHalo);
  TrySetPar(TracerParticleBitMask);

#undef TrySetPar

  if (ParameterName == "GroupParticleIdMask")
  {
    ParameterValue >> hex >> GroupParticleIdMask >> dec;
    cout << "GroupParticleIdMask = " << hex << GroupParticleIdMask << dec << endl;
    return true;
  }

  return false; // This signals to continue looking for valid parameter names
}

bool Parameter_t::TryMultipleValueParameter(string ParameterName, stringstream &ParameterValues)
{

  if (ParameterName == "SnapshotIdList")
  {
    for (int i; ParameterValues >> i;)
      SnapshotIdList.push_back(i);
    return true;
  }
  if (ParameterName == "TracerParticleTypes")
  {
    /* Store as a vector first to output in Params.log in a human-readable
     * format */
    TracerParticleTypes.clear(); // To remove default values
    for (int i; ParameterValues >> i;)
      TracerParticleTypes.push_back(i);

    /* Create a bitmask, to be used internally by the code to identify valid
     * tracer particle types*/
    TracerParticleBitMask = 0;
    for (int i : TracerParticleTypes)
      TracerParticleBitMask += 1 << i;
    return true;
  }
  return false; // This signals to continue looking for valid parameter names
}

void Parameter_t::SetParameterValue(const string &line)
{
  /* Get the name of the parameter */
  stringstream ss(line);
  string name;
  ss >> name;
  //   transform(name.begin(),name.end(),name.begin(),::tolower);

  /* We will try matching the name of the input parameter to those which have
   * been defined in the code. If we find a match, we assign its value. First
   * we try compulsory parameters, and then optional ones */
  if (TryCompulsoryParameterValue(name, ss))
    return;
  if (TrySingleValueParameter(name, ss))
    return;
  if (TryMultipleValueParameter(name, ss))
    return;

  /* No matching parameter name has been found, throw an error message */
  stringstream error_message;
  error_message << "Unrecognized configuration entry: " << name << endl;
  throw runtime_error(error_message.str());
}

void Parameter_t::ParseConfigFile(const char *param_file)
{
  ifstream ifs;
  ifs.open(param_file);
  if (!ifs.is_open()) // or ifs.fail()
  {
    cerr << "Error: failed to open configuration: " << param_file << endl;
    exit(1);
  }
  vector<string> lines;
  string line;

  cout << "Reading configuration file " << param_file << endl;

  while (getline(ifs, line))
  {
    trim_trailing_garbage(line, "#[");
    trim_leading_garbage(line, " \t");
    if (!line.empty())
      SetParameterValue(line);
  }
  CheckUnsetParameters();
  PhysicalConst::G = 43.0071 * (MassInMsunh / 1e10) / VelInKmS / VelInKmS / LengthInMpch;
  PhysicalConst::H0 = 100. * (1. / VelInKmS) / (1. / LengthInMpch);

  if (ParticleIdRankStyle)
    ParticleIdNeedHash = false;

  BoxHalf = BoxSize / 2.;
  TreeNodeResolution = SofteningHalo * 0.1;
  TreeNodeResolutionHalf = TreeNodeResolution / 2.;
  TreeNodeOpenAngleSquare = TreeNodeOpenAngle * TreeNodeOpenAngle;

  if (GroupFileFormat == "apostle_particle_index" || GroupFileFormat == "swiftsim_particle_index")
    GroupLoadedFullParticle = true;

  ReadSnapshotNameList();
}
void Parameter_t::ReadSnapshotNameList()
{ // to specify snapshotnamelist, create a file "snapshotlist.txt" under SubhaloPath, and list the filenames inside, one
  // per line.
  string snaplist_filename = SubhaloPath + "/snapshotlist.txt";
  ifstream ifs;
  ifs.open(snaplist_filename);
  if (ifs.is_open())
  {
    cout << "Found SnapshotNameList file " << snaplist_filename << endl;

    string line;
    while (getline(ifs, line))
    {
      trim_trailing_garbage(line, "#");
      istringstream ss(line);
      string name;
      ss >> name;
      if (!name.empty())
      {
        // 		cout<<name<<endl;
        SnapshotNameList.push_back(name);
      }
    }
  }
  if (SnapshotNameList.size())
    assert(SnapshotNameList.size() == MaxSnapshotIndex + 1);
}

void Parameter_t::CheckUnsetParameters()
{
  for (int i = 0; i < IsSet.size(); i++)
  {
    if (!IsSet[i])
    {
      cerr << "Error parsing configuration file: entry " << i << " missing\n";
      exit(1);
    }
  }
  if (!SnapshotIdList.empty())
  {
    int max_index = SnapshotIdList.size() - 1;
    if (MaxSnapshotIndex != max_index)
    {
      cerr << "Error: MaxSnapshotIndex=" << MaxSnapshotIndex << ", inconsistent with SnapshotIdList (" << max_index + 1
           << " snapshots listed)\n";
      exit(1);
    }
  }
}

void ParseHBTParams(int argc, char **argv, Parameter_t &config, int &snapshot_start, int &snapshot_end)
{
  if (argc < 2)
  {
    cerr << "Usage: " << argv[0] << " [param_file] <snapshot_start> <snapshot_end>\n";
    exit(1);
  }
  config.ParseConfigFile(argv[1]);
  if (2 == argc)
  {
    snapshot_start = config.MinSnapshotIndex;
    snapshot_end = config.MaxSnapshotIndex;
  }
  else
  {
    snapshot_start = atoi(argv[2]);
    if (argc > 3)
      snapshot_end = atoi(argv[3]);
    else
      snapshot_end = snapshot_start;
  }
  cout << "Running " << argv[0] << " from snapshot " << snapshot_start << " to " << snapshot_end
       << " using configuration file " << argv[1] << endl;
}

void Parameter_t::BroadCast(MpiWorker_t &world, int root)
/*sync parameters and physical consts across*/
{
#define _SyncVec(x, t) world.SyncContainer(x, t, root)
#define _SyncAtom(x, t) world.SyncAtom(x, t, root)
#define _SyncBool(x) world.SyncAtomBool(x, root)
#define _SyncVecBool(x) world.SyncVectorBool(x, root)
#define _SyncReal(x) _SyncAtom(x, MPI_HBT_REAL)

  _SyncVec(SnapshotPath, MPI_CHAR);
  _SyncVec(HaloPath, MPI_CHAR);
  _SyncVec(SubhaloPath, MPI_CHAR);
  _SyncVec(SnapshotFileBase, MPI_CHAR);
  _SyncAtom(MaxSnapshotIndex, MPI_INT);
  _SyncReal(BoxSize);
  _SyncReal(SofteningHalo);
  _SyncReal(MaxPhysicalSofteningHalo);
  _SyncVecBool(IsSet);

  _SyncVec(SnapshotDirBase, MPI_CHAR);
  _SyncVec(SnapshotFormat, MPI_CHAR);
  _SyncVec(GroupFileFormat, MPI_CHAR);
  _SyncAtom(MaxConcurrentIO, MPI_INT);
  _SyncAtom(MinSnapshotIndex, MPI_INT);
  _SyncAtom(MinNumPartOfSub, MPI_INT);
  _SyncAtom(GroupParticleIdMask, MPI_LONG);
  _SyncReal(MassInMsunh);
  _SyncReal(LengthInMpch);
  _SyncReal(VelInKmS);
  _SyncBool(PeriodicBoundaryOn);
  _SyncBool(SnapshotHasIdBlock);
  _SyncBool(ParticleIdRankStyle);
  _SyncBool(ParticleIdNeedHash);
  _SyncBool(SnapshotIdUnsigned);
  _SyncBool(SaveSubParticleProperties);
  _SyncBool(MergeTrappedSubhalos);
  _SyncVec(SnapshotIdList, MPI_INT);
  world.SyncVectorString(SnapshotNameList, root);

  _SyncReal(MajorProgenitorMassRatio);
#ifdef ALLOW_BINARY_SYSTEM
  _SyncReal(BinaryMassRatioLimit);
#endif
  _SyncReal(BoundMassPrecision);
  _SyncReal(SourceSubRelaxFactor);
  _SyncReal(SubCoreSizeFactor);
  _SyncAtom(SubCoreSizeMin, MPI_HBT_INT);

  _SyncReal(TreeAllocFactor);
  _SyncReal(TreeNodeOpenAngle);
  _SyncAtom(TreeMinNumOfCells, MPI_HBT_INT);

  _SyncAtom(MaxSampleSizeOfPotentialEstimate, MPI_HBT_INT);
  _SyncBool(RefineMostboundParticle);

  _SyncReal(TreeNodeOpenAngleSquare);
  _SyncReal(TreeNodeResolution);
  _SyncReal(TreeNodeResolutionHalf);
  _SyncReal(BoxHalf);

  _SyncBool(GroupLoadedFullParticle);
  _SyncAtom(TracerParticleBitMask, MPI_INT);
  //---------------end sync params-------------------------//

  _SyncReal(PhysicalConst::G);
  _SyncReal(PhysicalConst::H0);

#undef _SyncVec
#undef _SyncAtom
#undef _SyncBool
#undef _SyncVecBool
#undef _SyncReal
}
void Parameter_t::DumpParameters()
{
#ifndef HBT_VERSION
  cout << "Warning: HBT_VERSION unknown.\n";
#define HBT_VERSION "unknown"
#endif
  string filename = SubhaloPath + "/Parameters.log";
  ofstream version_file(filename, ios::out | ios::trunc);
  if (!version_file.is_open())
  {
    cerr << "Error opening " << filename << " for parameter dump." << endl;
    exit(1);
  }
  version_file << "#VERSION " << HBT_VERSION << endl;

#define DumpPar(var) version_file << #var << "  " << var << endl;
#define DumpComment(var)                                                                                               \
  {                                                                                                                    \
    version_file << "#";                                                                                               \
    DumpPar(var);                                                                                                      \
  }

  DumpPar(SnapshotPath);
  DumpPar(HaloPath);
  DumpPar(SubhaloPath);
  DumpPar(SnapshotFileBase);
  DumpPar(MaxSnapshotIndex) DumpPar(BoxSize);
  DumpPar(SofteningHalo);

  /*optional*/
  DumpPar(SnapshotDirBase);
  DumpPar(SnapshotFormat);
  DumpPar(GroupFileFormat);
  DumpPar(MaxConcurrentIO);
  DumpPar(MinSnapshotIndex);
  DumpPar(MinNumPartOfSub);
  DumpPar(MaxPhysicalSofteningHalo);

  if (GroupParticleIdMask)
    version_file << "GroupParticleIdMask " << hex << GroupParticleIdMask << dec << endl;

  DumpPar(MassInMsunh);
  DumpPar(LengthInMpch);
  DumpPar(VelInKmS);
  DumpPar(PeriodicBoundaryOn);
  DumpPar(SnapshotHasIdBlock);
  DumpPar(ParticleIdRankStyle);
  DumpPar(ParticleIdNeedHash);
  DumpPar(SnapshotIdUnsigned);
  DumpPar(SaveSubParticleProperties);
  DumpPar(MergeTrappedSubhalos);

  if (SnapshotIdList.size())
  {
    version_file << "SnapshotIdList";
    for (auto &&i : SnapshotIdList)
      version_file << " " << i;
    version_file << endl;
  }

  if (SnapshotNameList.size())
  {
    version_file << "#SnapshotNameList";
    for (auto &&i : SnapshotNameList)
      version_file << " " << i;
    version_file << endl;
  }

  if (TracerParticleTypes.size())
  {
    version_file << "TracerParticleTypes";
    for (auto &&i : TracerParticleTypes)
      version_file << " " << i;
    version_file << endl;
  }

  DumpPar(MajorProgenitorMassRatio);
  DumpPar(BoundMassPrecision);
  DumpPar(SourceSubRelaxFactor);
  DumpPar(SubCoreSizeFactor);
  DumpPar(SubCoreSizeMin);
  DumpPar(TreeAllocFactor);
  DumpPar(TreeNodeOpenAngle);
  DumpPar(TreeMinNumOfCells);
  DumpPar(MaxSampleSizeOfPotentialEstimate);
  DumpPar(RefineMostboundParticle);
  DumpComment(GroupLoadedFullParticle);

#undef DumpPar
#undef DumpComment
  version_file.close();
}

HBTReal Parameter_t::GetCurrentSoftening(HBTReal ScaleFactor)
{
  // Only one softening defined, use comoving
  if (MaxPhysicalSofteningHalo == -1)
    return SofteningHalo;

  // If two are defined, choose the one with the smallest value
  return min(SofteningHalo, MaxPhysicalSofteningHalo / ScaleFactor);
}
