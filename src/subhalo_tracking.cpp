#include <algorithm>
#include <iostream>
#include <new>
#include <omp.h>
#include <unordered_set>
#include <unordered_map>

#include "datatypes.h"
#include "snapshot_number.h"
#include "subhalo.h"

void Subhalo_t::UpdateTrack(const Snapshot_t &epoch)
{
  if (TrackId == SpecialConst::NullTrackId)
    return;

  if (0 == Rank)
    SnapshotIndexOfLastIsolation = epoch.GetSnapshotIndex();
  if (Mbound >= LastMaxMass)
  {
    SnapshotIndexOfLastMaxMass = epoch.GetSnapshotIndex();
    LastMaxMass = Mbound;
  }
}
HBTReal Subhalo_t::KineticDistance(const Halo_t &halo, const Snapshot_t &epoch)
{
  HBTxyz dv;
  epoch.RelativeVelocity(ComovingAveragePosition, PhysicalAverageVelocity, halo.ComovingAveragePosition,
                         halo.PhysicalAverageVelocity, dv);
  return VecNorm(dv);
}
void MemberShipTable_t::ResizeAllMembers(size_t n)
{
  auto olddata = AllMembers.data();
  AllMembers.resize(n);
  size_t offset = AllMembers.data() - olddata;
  if (offset)
  {
    for (HBTInt i = 0; i < Mem_SubGroups.size(); i++)
      Mem_SubGroups[i].Bind(Mem_SubGroups[i].data() + offset);
  }
}
void MemberShipTable_t::Init(const HBTInt nhalos, const HBTInt nsubhalos, const float alloc_factor)
{
  Mem_SubGroups.clear();
  Mem_SubGroups.resize(nhalos + 1);
  SubGroups.Bind(nhalos, Mem_SubGroups.data() + 1);

  AllMembers.clear();
  AllMembers.reserve(nsubhalos * alloc_factor); // allocate more for seed haloes.
  AllMembers.resize(nsubhalos);
}
void MemberShipTable_t::BindMemberLists()
{
  HBTInt offset = 0;
  for (HBTInt i = 0; i < Mem_SubGroups.size(); i++)
  {
    Mem_SubGroups[i].Bind(Mem_SubGroups[i].size(), &(AllMembers[offset]));
    offset += Mem_SubGroups[i].size();
    Mem_SubGroups[i].ReBind(0);
  }
}
void MemberShipTable_t::CountMembers(const SubhaloList_t &Subhalos, bool include_orphans)
{ // todo: parallelize this..
  if (include_orphans)
  {
    for (HBTInt subid = 0; subid < Subhalos.size(); subid++)
      SubGroups[Subhalos[subid].HostHaloId].IncrementBind();
  }
  else
  {
    for (HBTInt subid = 0; subid < Subhalos.size(); subid++)
      if (Subhalos[subid].Nbound > 1)
        SubGroups[Subhalos[subid].HostHaloId].IncrementBind();
  }
}
void MemberShipTable_t::FillMemberLists(const SubhaloList_t &Subhalos, bool include_orphans)
{ // fill with local subhaloid
  if (include_orphans)
  {
    for (HBTInt subid = 0; subid < Subhalos.size(); subid++)
    {
      /* We should have assigned a HostHaloId >= -1, if everything went well
       * during host deciding. */
      assert(Subhalos[subid].HostHaloId != -2);

      /* Populate FOFs with substructure. */
      SubGroups[Subhalos[subid].HostHaloId].PushBack(subid);
    }
  }
  else
  {
    for (HBTInt subid = 0; subid < Subhalos.size(); subid++)
      if (Subhalos[subid].Nbound > 1)
        SubGroups[Subhalos[subid].HostHaloId].PushBack(subid);
  }
}
struct CompareMass_t
{
  const SubhaloList_t *Subhalos;
  CompareMass_t(const SubhaloList_t &subhalos)
  {
    Subhalos = &subhalos;
  }
  bool operator()(const HBTInt &i, const HBTInt &j)
  {
    const Subhalo_t &sub1 = (*Subhalos)[i];
    const Subhalo_t &sub2 = (*Subhalos)[j];

    // Try to compare by mass first
    if (sub1.Mbound != sub2.Mbound)
      return sub1.Mbound > sub2.Mbound;

    // Where masses are equal, check vmax
    if (sub1.VmaxPhysical != sub2.VmaxPhysical)
      return sub1.VmaxPhysical > sub2.VmaxPhysical;

    // Where Vmax is equal try most bound ID
    if (sub1.MostBoundParticleId != sub2.MostBoundParticleId)
      return sub1.MostBoundParticleId > sub2.MostBoundParticleId;

    // Otherwise fall back to using the TrackId, which is always unique
    return sub1.TrackId > sub2.TrackId;
  }
};
void MemberShipTable_t::SortMemberLists(const SubhaloList_t &Subhalos)
{
  CompareMass_t compare_mass(Subhalos);
#pragma omp for
  for (HBTInt i = -1; i < SubGroups.size(); i++)
    std::sort(SubGroups[i].begin(), SubGroups[i].end(), compare_mass);
}
void MemberShipTable_t::SortSatellites(const SubhaloList_t &Subhalos)
/*central subhalo not changed*/
{
  CompareMass_t compare_mass(Subhalos);
  for (HBTInt i = 0; i < SubGroups.size(); i++)
    std::sort(SubGroups[i].begin() + 1, SubGroups[i].end(), compare_mass);
}
void MemberShipTable_t::AssignRanks(SubhaloList_t &Subhalos)
{
  // field subhaloes
  {
    MemberList_t &SubGroup = SubGroups[-1];
#pragma omp for
    for (HBTInt i = 0; i < SubGroup.size(); i++)
      Subhalos[SubGroup[i]].Rank = 0;
  }
#pragma omp for
  for (HBTInt haloid = 0; haloid < SubGroups.size(); haloid++)
  {
    MemberList_t &SubGroup = SubGroups[haloid];
    for (HBTInt i = 0; i < SubGroup.size(); i++)
      Subhalos[SubGroup[i]].Rank = i;
  }
}
void MemberShipTable_t::CountEmptyGroups()
{
  static HBTInt nbirth;
#pragma omp single
  nbirth = 0;
#pragma omp for reduction(+ : nbirth)
  for (HBTInt hostid = 0; hostid < SubGroups.size(); hostid++)
    if (SubGroups[hostid].size() == 0)
      nbirth++;
#pragma omp single
  NBirth = nbirth;
}
/*
inline bool SubhaloSnapshot_t::CompareHostAndMass(const HBTInt& subid_a, const HBTInt& subid_b)
{//ascending in host id, descending in mass inside each host, and put NullHaloId to the beginning.
  Subhalo_t a=Subhalos[subid_a], b=Subhalos[subid_b];

  if(a.HostHaloId==b.HostHaloId) return (a.Nbound>b.Nbound);

  return (a.HostHaloId<b.HostHaloId); //(a.HostHaloId!=SpecialConst::NullHaloId)&&
}*/
void MemberShipTable_t::Build(const HBTInt nhalos, const SubhaloList_t &Subhalos, bool include_orphans)
{
#pragma omp single
  {
    Init(nhalos, Subhalos.size());
    CountMembers(Subhalos, include_orphans);
    BindMemberLists();
    FillMemberLists(Subhalos, include_orphans);
  }
  SortMemberLists(Subhalos);
  CountEmptyGroups();
  //   std::sort(AllMembers.begin(), AllMembers.end(), CompareHostAndMass);
}

inline HBTInt GetLocalHostId(HBTInt pid, const HaloSnapshot_t &halo_snap, const ParticleSnapshot_t &part_snap)
{
  HBTInt hostid = halo_snap.ParticleHash.GetIndex(pid);
  if (hostid < 0) // not in the haloes, =-1
  {
    if (part_snap.GetIndex(pid) == SpecialConst::NullParticleId)
      hostid--; // not in this snapshot either, =-2
  }
  return hostid;
}

/* Identify and store the particle IDs of the most bound NumTracerHostFinding
 * collisionless tracers. Used to identify host FOF groups. */
void GetTracerIds(vector<HBTInt>::iterator particle_ids, const Subhalo_t &Subhalo)
{
  /* Initialise vector. This will make it so the code knows when to stop looking
   * for tracers, since orphans will have all but the first with NullParticleId */
  fill(particle_ids, particle_ids + HBTConfig.NumTracerHostFinding, SpecialConst::NullParticleId);

  /* Handle orphans by manually copying over the tracer particle ID, since we do
   * not have any particles associated to them. */
  if (!Subhalo.Particles.size())
  {
    particle_ids[0] = Subhalo.MostBoundParticleId;
    return;
  }

  /* Iterate over the particle list to find tracers. */
  int BoundRanking = 0;
  for (auto particle = Subhalo.Particles.begin(); particle != Subhalo.Particles.begin() + Subhalo.Nbound; particle++)
  {
    if (particle->IsTracer()) // Only use tracers.
      particle_ids[BoundRanking++] = particle->Id;

    if (BoundRanking == HBTConfig.NumTracerHostFinding) // Use up to a user-defined number of tracers.
      break;
  }

  /* Sanity checks */
  assert(BoundRanking == min((int)Subhalo.Particles.size(),
                             HBTConfig.NumTracerHostFinding)); // We found all expected particles
}

/* Identify which FOF is host to the particles. If we have a value of -2, that
 * means the particle was not found in the particle information available to the
 * local rank. */
bool GetTracerHosts(vector<HBTInt>::iterator particle_hosts, vector<HBTInt>::const_iterator particle_ids,
                    const HaloSnapshot_t &halo_snap, const ParticleSnapshot_t &part_snap)
{
  /* Initialise vector. This will make it so the code knows when to stop looking
   * for tracers, since orphans will have all but the first with NullParticleId */
  fill(particle_hosts, particle_hosts + HBTConfig.NumTracerHostFinding, -2);

  bool MakeDecision = true;

  /* Iterate over the particle list to find hosts. */
  for (int BoundRanking = 0; BoundRanking < HBTConfig.NumTracerHostFinding; BoundRanking++)
  {
    // No more tracers left (should only occur for orphans!)
    if (particle_ids[BoundRanking] == SpecialConst::NullParticleId)
      break;

    // Get host, which can be -1
    particle_hosts[BoundRanking] = halo_snap.ParticleHash.GetIndex(particle_ids[BoundRanking]);

    /* We cannot find the particle in the current rank. We will therefore need to
     * try to find it in other ranks (and hence delay making a host decision) */
    if (particle_hosts[BoundRanking] == -1)
    {
      MakeDecision = false; // Defer making a decision until we have all info.
      if (part_snap.GetIndex(particle_ids[BoundRanking]) == SpecialConst::NullParticleId)
        particle_hosts[BoundRanking]--; // Turns it to -2
    }
  }

  return MakeDecision;
}

/* Assign a host to a given subgroup based on the FoF membership of the
 * NumTracerHostFinding most bound collisionless particles. This decision is
 * weighted by the binding energy ranking the particles had in the previous
 * output. */
HBTInt DecideLocalHostId(vector<HBTInt>::const_iterator particle_hosts)
{
  /* To store unique host candidates, and the matching score. */
  unordered_map<HBTInt, float> CandidateHosts;

  /* Iterate over the particle list, and weight each candidate score by how
   * bound was the particle is */
  for (int BoundRanking = 0; BoundRanking < HBTConfig.NumTracerHostFinding; BoundRanking++)
  {
    // Not a valid tracer (NOTE: should we break or continue here?)
    if (particle_hosts[BoundRanking] == -2)
      continue;

    /* If the key is not present, the score gets zero-initialised by default */
    CandidateHosts[particle_hosts[BoundRanking]] += 1.0 / (1 + pow(float(BoundRanking), 0.5));
  }

  /* Select candidate host with the highest score. */
  HBTInt HostId = -2; // Default value
  float MaximumScore = 0;

  for (auto candidate : CandidateHosts)
  {
    if (candidate.second > MaximumScore)
    {
      HostId = candidate.first;
      MaximumScore = candidate.second;
    }
  }

  return HostId;
}

/* Assign a host to a given subgroup based on the FoF membership of the
 * NumTracerHostFinding most bound collisionless particles. This decision is
 * weighted by the binding energy ranking the particles had in the previous
 * output. Used during search across multiple tasks.*/
IdRank_t DecideLocalHostId(vector<IdRank_t>::const_iterator particle_hosts)
{
  /* To store unique host candidates, their rank, and the matching score. */
  unordered_map<HBTInt, float> CandidateHosts, CandidateHostRank;

  /* Iterate over the particle list, and weight each candidate score by how
   * bound was the particle is */
  for (int BoundRanking = 0; BoundRanking < HBTConfig.NumTracerHostFinding; BoundRanking++)
  {
    // Not a valid tracer (NOTE: should we break or continue here?)
    if (particle_hosts[BoundRanking].Id == -2)
      continue;

    /* If the key is not present, the score gets zero-initialised by default */
    CandidateHosts[particle_hosts[BoundRanking].Id] += 1.0 / (1 + pow(float(BoundRanking), 0.5));

    /* If we have hostless particles, we assign the rank to be the one where the
     * most bound hostless particle is present. */
    if (particle_hosts[BoundRanking].Id == -1)
    {
      auto findit = CandidateHostRank.find(particle_hosts[BoundRanking].Id);
      if (findit == CandidateHostRank.end()) // First time we find a hostless particle.
        CandidateHostRank[particle_hosts[BoundRanking].Id] = particle_hosts[BoundRanking].Rank;
    }

    // Store its rank
    CandidateHostRank[particle_hosts[BoundRanking].Id] = particle_hosts[BoundRanking].Rank;
  }

  /* Select candidate host with the highest score. */
  IdRank_t HostIdRank{-2, -1};
  float MaximumScore = 0;

  for (auto candidate : CandidateHosts)
  {
    if (candidate.second > MaximumScore)
    {
      HostIdRank.Id = candidate.first;
      MaximumScore = candidate.second;
    }
  }

  /* Assign target rank where the subhalo will be moved to. */
  HostIdRank.Rank = CandidateHostRank[HostIdRank.Id];

  return HostIdRank;
}

void FindLocalHosts(const HaloSnapshot_t &halo_snap, const ParticleSnapshot_t &part_snap, vector<Subhalo_t> &Subhalos,
                    vector<Subhalo_t> &LocalSubhalos)
{
#pragma omp parallel for
  for (HBTInt subid = 0; subid < Subhalos.size(); subid++)
  {
    /* Create a list of tracer particle IDs*/
    vector<HBTInt> TracerParticleIds(HBTConfig.NumTracerHostFinding);
    GetTracerIds(TracerParticleIds.begin(), Subhalos[subid]);

    /* Identify which FOFs those IDs are located in. */
    vector<HBTInt> TracerHosts(HBTConfig.NumTracerHostFinding);
    bool MakeDecision = GetTracerHosts(TracerHosts.begin(), TracerParticleIds.cbegin(), halo_snap, part_snap);

    /* If we found all tracers in the current rank, make a host decision.
     * Otherwise, we will need to use information from other ranks to be sure
     * of where the track is. */
    Subhalos[subid].HostHaloId = (MakeDecision) ? DecideLocalHostId(TracerHosts.cbegin()) : -2;
  }

  HBTInt nsub = 0;
  for (HBTInt subid = 0; subid < Subhalos.size(); subid++)
  {
    if (Subhalos[subid].HostHaloId < 0) // Move all subhaloes
    {
      if (subid > nsub)
        Subhalos[nsub] = move(Subhalos[subid]); // there should be a default move assignement operator.
      nsub++;
    }
    else
      LocalSubhalos.push_back(move(Subhalos[subid]));
  }
  Subhalos.resize(nsub);
}

void FindOtherHosts(MpiWorker_t &world, int root, const HaloSnapshot_t &halo_snap, const ParticleSnapshot_t &part_snap,
                    VectorView_t<Subhalo_t> &Subhalos, vector<Subhalo_t> &LocalSubhalos,
                    MPI_Datatype MPI_Subhalo_Shell_Type)
/*scatter Subhalos from process root to LocalSubhalos in every other process
 Note Subalos are "moved", so are in a unspecified state upon return.*/
{
  int thisrank = world.rank();
  vector<HBTInt> TrackParticleIds;
  HBTInt NumSubhalos;

  // broadcast trackparticles
  if (thisrank == root)
  {
    NumSubhalos = Subhalos.size();
    if (NumSubhalos * HBTConfig.NumTracerHostFinding > INT_MAX)
      throw runtime_error("Error: in FindOtherHosts(), sending more subhaloes than INT_MAX will cause MPI message to "
                          "overflow. Please try more MPI threads. aborting.\n");
  }
  MPI_Bcast(&NumSubhalos, 1, MPI_HBT_INT, root, world.Communicator);

  /* Create vector to hold the IDs of particles to look out for. We assign a
   * conservative value of NumTracerHostFinding particles per subhalo, even
   * though we may have orphans in the mix (only require one entry) */
  vector<HBTInt> TracerParticleIds(NumSubhalos * HBTConfig.NumTracerHostFinding);

  /* Populate the vectors with the ParticleIDs of tracers belonging to the subhaloes
   * of the root task */
  if (thisrank == root)
  {
#pragma omp parallel for if (NumSubhalos > 20)
    for (HBTInt i = 0; i < NumSubhalos; i++)
      GetTracerIds(TracerParticleIds.begin() + i * HBTConfig.NumTracerHostFinding, Subhalos[i]);
  }

  /* Tell other tasks which IDs to look out for. */
  MPI_Bcast(TracerParticleIds.data(), TracerParticleIds.size(), MPI_HBT_INT, root, world.Communicator);

  /* To find hosts in the current task  */
  vector<IdRank_t> LocalHostRankPairs(NumSubhalos * HBTConfig.NumTracerHostFinding, IdRank_t{-2, thisrank});

#pragma omp parallel for if (NumSubhalos > 20)
  for (HBTInt i = 0; i < NumSubhalos; i++)
  {
    unsigned long long offset = HBTConfig.NumTracerHostFinding * i; // Start of subhalo

    /* Identify which FOFs those IDs are located in. */
    vector<HBTInt> TracerHosts(HBTConfig.NumTracerHostFinding);
    GetTracerHosts(TracerHosts.begin(), TracerParticleIds.cbegin() + offset, halo_snap, part_snap);

    /* Copy over to the vector-struct */
    for (int BoundRanking = 0; BoundRanking < HBTConfig.NumTracerHostFinding; BoundRanking++)
      LocalHostRankPairs[offset + BoundRanking].Id = TracerHosts[BoundRanking];
  }

  /* Communicate to all tasks */
  vector<IdRank_t> GlobalHostRankPairs(NumSubhalos * HBTConfig.NumTracerHostFinding);
  MPI_Allreduce(LocalHostRankPairs.data(), GlobalHostRankPairs.data(), LocalHostRankPairs.size(), MPI_HBTRankPair,
                MPI_MAXLOC, world.Communicator);

  /* Score each candidate, and identify the rank it lives in */
  vector<IdRank_t> GlobalHostIds(NumSubhalos);
#pragma omp parallel for if (NumSubhalos > 20)
  for (HBTInt i = 0; i < NumSubhalos; i++)
  {
    unsigned long long offset = HBTConfig.NumTracerHostFinding * i; // Start of subhalo
    GlobalHostIds[i] = DecideLocalHostId(GlobalHostRankPairs.cbegin() + offset);
  }

  // scatter free subhaloes from root to everywhere
  // send particles and nests; no scatterw, do it manually
  MPI_Datatype MPI_HBT_Particle;
  Particle_t().create_MPI_type(MPI_HBT_Particle);
  vector<vector<int>> SendSizes(world.size()), SendNestSizes(world.size());
  vector<MPI_Request> Req0, Req1, ReqNest0, ReqNest1;
  if (thisrank == root)
  {
    vector<vector<MPI_Aint>> SendBuffers(world.size()), SendNestBuffers(world.size());
    for (HBTInt subid = 0; subid < NumSubhalos; subid++) // packing
    {
      int rank = GlobalHostIds[subid].Rank;
      auto &Particles = Subhalos[subid].Particles;
      MPI_Aint p;
      MPI_Get_address(Particles.data(), &p);
      SendBuffers[rank].push_back(p);
      SendSizes[rank].push_back(Particles.size());

      auto &Nest = Subhalos[subid].NestedSubhalos;
      MPI_Get_address(Nest.data(), &p);
      SendNestBuffers[rank].push_back(p);
      SendNestSizes[rank].push_back(Nest.size());
    }
    Req0.resize(world.size());
    Req1.resize(world.size());
    ReqNest0.resize(world.size());
    ReqNest1.resize(world.size());
    for (int rank = 0; rank < world.size(); rank++)
    {
      {
        MPI_Isend(SendSizes[rank].data(), SendSizes[rank].size(), MPI_INT, rank, 0, world.Communicator, &Req0[rank]);
        MPI_Datatype SendType;
        MPI_Type_create_hindexed(SendSizes[rank].size(), SendSizes[rank].data(), SendBuffers[rank].data(),
                                 MPI_HBT_Particle, &SendType);
        MPI_Type_commit(&SendType);
        MPI_Isend(MPI_BOTTOM, 1, SendType, rank, 1, world.Communicator, &Req1[rank]);
        MPI_Type_free(&SendType);
      }
      {
        MPI_Isend(SendNestSizes[rank].data(), SendNestSizes[rank].size(), MPI_INT, rank, 0, world.Communicator,
                  &ReqNest0[rank]);
        MPI_Datatype SendNestType;
        MPI_Type_create_hindexed(SendNestSizes[rank].size(), SendNestSizes[rank].data(), SendNestBuffers[rank].data(),
                                 MPI_HBT_INT, &SendNestType);
        MPI_Type_commit(&SendNestType);
        MPI_Isend(MPI_BOTTOM, 1, SendNestType, rank, 1, world.Communicator, &ReqNest1[rank]);
        MPI_Type_free(&SendNestType);
      }
    }
  }
  // receive on every process, including root
  vector<MPI_Aint> ReceiveBuffer, ReceiveNestBuffer;
  vector<int> ReceiveSize, ReceiveNestSize;
  int NumNewSubs;
  MPI_Status stat;
  MPI_Probe(root, 0, world.Communicator, &stat);
  MPI_Get_count(&stat, MPI_INT, &NumNewSubs);
  ReceiveSize.resize(NumNewSubs);
  MPI_Recv(ReceiveSize.data(), NumNewSubs, MPI_INT, root, 0, world.Communicator, &stat);
  LocalSubhalos.resize(LocalSubhalos.size() + NumNewSubs);
  auto NewSubhalos = LocalSubhalos.end() - NumNewSubs;
  ReceiveBuffer.resize(NumNewSubs);
  for (int i = 0; i < NumNewSubs; i++)
  {
    auto &Particles = NewSubhalos[i].Particles;
    Particles.resize(ReceiveSize[i]);
    MPI_Aint p;
    MPI_Get_address(Particles.data(), &p);
    ReceiveBuffer[i] = p;
  }
  MPI_Datatype ReceiveType;
  MPI_Type_create_hindexed(NumNewSubs, ReceiveSize.data(), ReceiveBuffer.data(), MPI_HBT_Particle, &ReceiveType);
  MPI_Type_commit(&ReceiveType);
  MPI_Recv(MPI_BOTTOM, 1, ReceiveType, root, 1, world.Communicator, &stat);
  MPI_Type_free(&ReceiveType);

  MPI_Type_free(&MPI_HBT_Particle);

  ReceiveNestSize.resize(NumNewSubs);
  MPI_Recv(ReceiveNestSize.data(), NumNewSubs, MPI_INT, root, 0, world.Communicator, &stat);
  ReceiveNestBuffer.resize(NumNewSubs);
  for (int i = 0; i < NumNewSubs; i++)
  {
    auto &Nest = NewSubhalos[i].NestedSubhalos;
    Nest.resize(ReceiveNestSize[i]);
    MPI_Aint p;
    MPI_Get_address(Nest.data(), &p);
    ReceiveNestBuffer[i] = p;
  }
  MPI_Datatype ReceiveNestType;
  MPI_Type_create_hindexed(NumNewSubs, ReceiveNestSize.data(), ReceiveNestBuffer.data(), MPI_HBT_INT, &ReceiveNestType);
  MPI_Type_commit(&ReceiveNestType);
  MPI_Recv(MPI_BOTTOM, 1, ReceiveNestType, root, 1, world.Communicator, &stat);
  MPI_Type_free(&ReceiveNestType);

  if (thisrank == root)
  {
    // 	  vector <MPI_Status> stats(world.size());
    MPI_Waitall(world.size(), Req0.data(), MPI_STATUS_IGNORE);
    MPI_Waitall(world.size(), Req1.data(), MPI_STATUS_IGNORE);
    MPI_Waitall(world.size(), ReqNest0.data(), MPI_STATUS_IGNORE);
    MPI_Waitall(world.size(), ReqNest1.data(), MPI_STATUS_IGNORE);
  }

  // copy other properties
  vector<int> Counts(world.size()), Disps(world.size());
  vector<Subhalo_t> TmpHalos;
  if (world.rank() == root)
  { // reuse GlobalHostIds for sorting
    for (HBTInt subid = 0; subid < Subhalos.size(); subid++)
    {
      Subhalos[subid].HostHaloId = GlobalHostIds[subid].Id;
      // 		assert(GlobalHostIds[subid].n>=-1);
      GlobalHostIds[subid].Id = subid;
    }
    stable_sort(GlobalHostIds.begin(), GlobalHostIds.end(), CompareRank);
    TmpHalos.resize(Subhalos.size());
    for (HBTInt subid = 0; subid < Subhalos.size(); subid++)
      TmpHalos[subid] = move(Subhalos[GlobalHostIds[subid].Id]);
    // 		Subhalos[GlobalHostIds[subid].n].MoveTo(TmpHalos[subid], false);
    for (int rank = 0; rank < world.size(); rank++)
      Counts[rank] = SendSizes[rank].size();
    CompileOffsets(Counts, Disps);
  }
  MPI_Scatterv(TmpHalos.data(), Counts.data(), Disps.data(), MPI_Subhalo_Shell_Type, &NewSubhalos[0], NumNewSubs,
               MPI_Subhalo_Shell_Type, root, world.Communicator);
}
void FindOtherHostsSafely(MpiWorker_t &world, int root, const HaloSnapshot_t &halo_snap,
                          const ParticleSnapshot_t &part_snap, vector<Subhalo_t> &Subhalos,
                          vector<Subhalo_t> &LocalSubhalos, MPI_Datatype MPI_Subhalo_Shell_Type)
/*break Subhalos into small chunks and then FindOtherHosts() for them, to avoid overflow in MPI message size*/
{
  const int MaxChunkSize = 1024 * 1024;
  int flagstop = 0;
  VectorView_t<Subhalo_t> HaloChunk;
  if (world.rank() == root)
  {
    HBTInt offset = 0, chunksize = 0;
    for (HBTInt i = 0; i < Subhalos.size(); i++)
    {
      chunksize += Subhalos[i].Particles.size();
      if (chunksize >= MaxChunkSize)
      { // send buffer
        HaloChunk.Bind(i - offset, &Subhalos[offset]);
        if (HaloChunk.size())
        {
          MPI_Bcast(&flagstop, 1, MPI_INT, root, world.Communicator);
          FindOtherHosts(world, root, halo_snap, part_snap, HaloChunk, LocalSubhalos, MPI_Subhalo_Shell_Type);
        }
        // reset buffer
        offset = i;
        chunksize = Subhalos[i].Particles.size(); // the current halo
      }
    }
    // remaining ones
    if (Subhalos.size() > offset)
    {
      HaloChunk.Bind(Subhalos.size() - offset, &Subhalos[offset]);
      MPI_Bcast(&flagstop, 1, MPI_INT, root, world.Communicator);
      FindOtherHosts(world, root, halo_snap, part_snap, HaloChunk, LocalSubhalos, MPI_Subhalo_Shell_Type);
    }
    flagstop = 1;
    MPI_Bcast(&flagstop, 1, MPI_INT, root, world.Communicator);
  }
  else
  {
    while (true)
    {
      MPI_Bcast(&flagstop, 1, MPI_INT, root, world.Communicator);
      if (flagstop)
        break;
      else
        FindOtherHosts(world, root, halo_snap, part_snap, HaloChunk, LocalSubhalos, MPI_Subhalo_Shell_Type);
    }
  }
}
void SubhaloSnapshot_t::AssignHosts(MpiWorker_t &world, HaloSnapshot_t &halo_snap, const ParticleSnapshot_t &part_snap)
/* find host haloes for subhaloes, and build MemberTable. Each subhalo is moved to the processor of its host halo, with
 * its HostHaloId set to the local haloid of the host*/
{
  ParallelizeHaloes = halo_snap.NumPartOfLargestHalo < 0.1 * halo_snap.TotNumberOfParticles; // no dominating objects

  /* To hold subhaloes with updated information. */
  vector<Subhalo_t> LocalSubhalos;
  LocalSubhalos.reserve(Subhalos.size());

  /* Creates ParticleID - HostHalo information, local to the task. */
  halo_snap.FillParticleHash();

  /* Try identifying which FOF hosts local subhaloes, using local information
   * only. */
  FindLocalHosts(halo_snap, part_snap, Subhalos, LocalSubhalos);

  /* Those which require information from external ranks are dealt with here. */
  for (int rank = 0; rank < world.size(); rank++)
    FindOtherHostsSafely(world, rank, halo_snap, part_snap, Subhalos, LocalSubhalos, MPI_HBT_SubhaloShell_t);
  
  /* Here we will change the host halo ID of subhalos whose tracers were lost*/
  HandleTracerlessSubhalos(world, LocalSubhalos);

  Subhalos.swap(LocalSubhalos);
  halo_snap.ClearParticleHash();

  MemberTable.Build(halo_snap.Halos.size(), Subhalos, true);
}

/* This function iterates over Subhalos and assigns a default HostHaloId (-1) to all Subhalos
 * whose tracer particles were not found. If any such cases are present in the simulation, a 
 * warning is printed. */
void SubhaloSnapshot_t::HandleTracerlessSubhalos(MpiWorker_t &world, vector<Subhalo_t> &LocalSubhalos)
{
  HBTInt NumberTracerlessSubhalos = 0;

#pragma omp parallel for reduction (+:NumberTracerlessSubhalos) if (LocalSubhalos.size() > 20)
  for (HBTInt i = 0; i < LocalSubhalos.size(); i++)
  {
    if(LocalSubhalos[i].HostHaloId == -2) // Tracerless
    {
      LocalSubhalos[i].HostHaloId = -1; // Make them hostless
      NumberTracerlessSubhalos++;
    }
  }

  /* Determine if any subgroup in the simulation had missing tracers, and if so,
   * print out a warning */
  HBTInt GlobalNumberTracelessSubhalos = 0;
  MPI_Allreduce(&NumberTracerlessSubhalos, &GlobalNumberTracelessSubhalos, 1, MPI_HBT_INT, MPI_SUM, world.Communicator);

  if((GlobalNumberTracelessSubhalos > 0) && (world.rank() == 0))
      std::cout << "WARNING: " << GlobalNumberTracelessSubhalos << " subhalos were missing their particle tracers. This is likely due to using particles that dissapear from the simulation (e.g. merging black holes) as subahalo tracers." << std::endl;
}

/* Constrains subhaloes to only exist within a single host. This prevents
 * duplications from occuring as a result of this. For example, a particle
 * associated to a satellite subhalo that is in a different FOF, hence being
 * fed to its corresponding central. */
void SubhaloSnapshot_t::ConstrainToSingleHost(const HaloSnapshot_t &halo_snap)
{
  /* Remove particles assigned to a host different to the one assigned to the
   * subhalo. Need to pass entry from halo_snap.Halos, because HostHaloId in
   * subhalos is local value, rather than global. */
#pragma omp parallel for schedule(dynamic, 1) if (ParallelizeHaloes)
  for (HBTInt subid = 0; subid < Subhalos.size(); subid++)
  {
    /* Need to be careful with hostless haloes, as otherwise we try to access
     * entry -1 */
    HBTInt GlobalHostHaloId = (Subhalos[subid].HostHaloId == -1) ? HBTConfig.ParticleNullGroupId
                                                                 : halo_snap.Halos[Subhalos[subid].HostHaloId].HaloId;
    Subhalos[subid].RemoveOtherHostParticles(GlobalHostHaloId);
  }
}

/* This will remove from the source of the current subhalo all particles that
 * belong to a FOF different to the one formally assigned to it, unless the
 * particle is hostless. We do not need to worry about orphans; they will always
 * be in the host of its tracer at this point. Centrals get applied this
 * function (since current depth value is not reflective of hierarchy) but the
 * particles get swapped with those of the FOF anyway. */
void Subhalo_t::RemoveOtherHostParticles(const HBTInt &GlobalHostHaloId)
{
  /* Criteria used to flag particles to remove. We use NullGroupId rather than -1,
   * since the value is input dependent. */
  auto CheckHostMembership = [&](Particle_t const &particle) {
    return (particle.HostId != (GlobalHostHaloId)) && (particle.HostId != HBTConfig.ParticleNullGroupId);
  };

  /* Identify (FOF-hosted) particles not present in the one assigned to the
   * subgroup. */
  auto foreign_particles = std::remove_if(Particles.begin(), Particles.end(), CheckHostMembership);

  /* Remove from vector */
  Particles.erase(foreign_particles, Particles.end());

  /* Update number of particles in the source */
  Nbound = Particles.size();
}

void SubhaloSnapshot_t::DecideCentrals(const HaloSnapshot_t &halo_snap)
/* to select central subhalo according to KineticDistance, and move each central to the beginning of each list in
 * MemberTable*/
{
#pragma omp for
  for (HBTInt hostid = 0; hostid < halo_snap.Halos.size(); hostid++)
  {
    MemberShipTable_t::MemberList_t &List = MemberTable.SubGroups[hostid];
    if (List.size() > 1)
    {
      int n_major;
      HBTReal MassLimit = Subhalos[List[0]].Mbound * HBTConfig.MajorProgenitorMassRatio;
      for (n_major = 1; n_major < List.size(); n_major++)
        if (Subhalos[List[n_major]].Mbound < MassLimit)
          break;
      if (n_major > 1)
      {
        HBTReal dmin = Subhalos[List[0]].KineticDistance(halo_snap.Halos[hostid], *this);
        int icenter = 0;
        for (int i = 1; i < n_major; i++)
        {
          HBTReal d = Subhalos[List[i]].KineticDistance(halo_snap.Halos[hostid], *this);
          if (dmin > d)
          {
            dmin = d;
            icenter = i;
          }
        }
        if (icenter)
          swap(List[0], List[icenter]);
      }
    }
  }
}

void SubhaloSnapshot_t::FeedCentrals(HaloSnapshot_t &halo_snap)
/* replace centrals with host particles;
 * create a new central if there is none
 * initialize new central with host halo center coordinates
 * halo_snap is rendered unspecified upon return (its particles have been swapped to subhaloes).
 */
{
  static HBTInt Npro;
#pragma omp single
  {
    Npro = Subhalos.size();
    // Subhalos.reserve(Snapshot->size()*0.1);//reserve enough	branches.......
    Subhalos.resize(Npro + MemberTable.NBirth);
  }
#pragma omp for
  for (HBTInt hostid = 0; hostid < halo_snap.Halos.size(); hostid++)
  {
    MemberShipTable_t::MemberList_t &Members = MemberTable.SubGroups[hostid];
    auto &Host = halo_snap.Halos[hostid];
    if (0 == Members.size()) // create a new sub
    {
      HBTInt subid;
#pragma omp critical(AddNewSub) // maybe consider ordered for here..
      {
        subid = Npro++;
      }
      auto &central = Subhalos[subid];
      central.HostHaloId = hostid;
      copyHBTxyz(central.ComovingAveragePosition, Host.ComovingAveragePosition);
      copyHBTxyz(central.PhysicalAverageVelocity, Host.PhysicalAverageVelocity);
      central.Particles.swap(Host.Particles);
      central.Nbound = central.Particles.size(); // init Nbound to source size.
      central.SnapshotIndexOfBirth = SnapshotIndex;
    }
    else
    {
      auto &central = Subhalos[Members[0]];

      // Only test whether we have particles for previously-resolved subhaloes
      if (central.IsAlive())
        assert(central.Particles.size());

      /* The subhalo now contains all the host particles. Those belonging to
       * subhaloes will be masked during unbinding, if exclusive mass option is
       * used. */
      central.Particles.swap(Host.Particles);
      central.Nbound = central.Particles.size();
    }
  }
  //   #pragma omp single
  //   halo_snap.Clear();//to avoid misuse
}
void SubhaloSnapshot_t::PrepareCentrals(MpiWorker_t &world, HaloSnapshot_t &halo_snap)
{
#pragma omp parallel
  {
    DecideCentrals(halo_snap);
    FeedCentrals(halo_snap);
  }
  NestSubhalos(world);
#ifndef INCLUSIVE_MASS
  MaskSubhalos();
#endif
}

void SubhaloSnapshot_t::RegisterNewTracks(MpiWorker_t &world)
/*assign trackId to new bound ones, remove unbound ones, and rebuild membership*/
{
  HBTInt NumSubMax = Subhalos.size(), NumSubOld = NumSubMax - MemberTable.NBirth;
  HBTInt NumSubNew = NumSubOld;
  for (HBTInt i = NumSubNew; i < NumSubMax; i++)
  {
    if (Subhalos[i].Nbound > 1)
    {
      if (i != NumSubNew)
        Subhalos[NumSubNew] = move(Subhalos[i]);
      NumSubNew++;
    }
  }
  Subhalos.resize(NumSubNew);
  MemberTable.Build(MemberTable.SubGroups.size(), Subhalos,
                    true); // rebuild membership with new subs and also include orphans this time.
  MemberTable.NFake = NumSubMax - NumSubNew;
  MemberTable.NBirth = NumSubNew - NumSubOld;

  // now assign a global TrackId
  HBTInt TrackIdOffset, NBirth = MemberTable.NBirth, GlobalNumberOfSubs;
  MPI_Allreduce(&NumSubOld, &GlobalNumberOfSubs, 1, MPI_HBT_INT, MPI_SUM, world.Communicator);
  MPI_Scan(&NBirth, &TrackIdOffset, 1, MPI_HBT_INT, MPI_SUM, world.Communicator);
  TrackIdOffset = TrackIdOffset + GlobalNumberOfSubs - NBirth;
  for (HBTInt i = NumSubOld; i < NumSubNew; i++)
    Subhalos[i].TrackId = TrackIdOffset++;
}
void SubhaloSnapshot_t::PurgeMostBoundParticles()
/* fix the possible issue that the most-bound particle of a subhalo might belong to a sub-sub.
 * this is achieved by masking particles from smaller subs and promote the remaining most-bound particle in each
 * subhalo. orphan galaxies are not considered. the shift in center position does not affect the total angular momentum.
 */
{
#pragma omp for
  for (HBTInt i = -1; i < MemberTable.SubGroups.size(); i++)
  {
    auto &Group = MemberTable.SubGroups[i];
    unordered_set<HBTInt> ExclusionList;
    {
      HBTInt np = 0;
      for (auto &&subid : Group)
        if (Subhalos[subid].Nbound > 1)
          np += Subhalos[subid].Nbound;
      ExclusionList.reserve(np);
    }
    for (HBTInt j = Group.size() - 1; j >= 0; j--)
    {
      auto &subhalo = Subhalos[Group[j]];
      if (subhalo.Nbound > 1)
      {
        for (auto &p : subhalo.Particles)
        {
          if (ExclusionList.find(p.Id) == ExclusionList.end())
          {
            if (&p != &subhalo.Particles[0])
            {
              copyHBTxyz(subhalo.ComovingMostBoundPosition, p.ComovingPosition);
              copyHBTxyz(subhalo.PhysicalMostBoundVelocity, p.GetPhysicalVelocity());
              swap(subhalo.Particles[0], p);
              break;
            }
          }
        }
        for (auto &&p : subhalo.Particles)
          ExclusionList.insert(p.Id); // alternative: only filter most-bounds
      }
    }
  }
}

void SubhaloSnapshot_t::SetNestedParentIds()
{

  TrackKeyList_t Ids(*this);
  MappedIndexTable_t<HBTInt, HBTInt> TrackHash;
  TrackHash.Fill(Ids, SpecialConst::NullTrackId);

  // Initialize all subhalos to no parent
  for (auto &&subhalo : Subhalos)
    subhalo.NestedParentTrackId = SpecialConst::NullTrackId;

  // Use nesting info to set parent trackid
  for (auto &&subhalo : Subhalos)
  {
    for (auto &nested_trackid : subhalo.NestedSubhalos)
    {
      HBTInt child_index = TrackHash.GetIndex(nested_trackid);
      Subhalos[child_index].NestedParentTrackId = subhalo.TrackId;
    }
  }
}

void SubhaloSnapshot_t::LocalizeNestedIds(MpiWorker_t &world)
/*convert TrackIds of NestedSubhalos to local index, and move non-local nestedsubhalos to their host processors as
 * DissociatedTrack*/
{
  TrackKeyList_t Ids(*this);
  MappedIndexTable_t<HBTInt, HBTInt> TrackHash;
  TrackHash.Fill(Ids, SpecialConst::NullTrackId);

  // collect lost tracks
  vector<HBTInt> DissociatedTracks;
  for (auto &&subhalo : Subhalos)
  {
    auto &nests = subhalo.NestedSubhalos;
    auto it_begin = nests.begin();
    auto it_save = it_begin, it = it_begin;
    for (; it != nests.end(); ++it)
    {
      HBTInt subid = TrackHash.GetIndex(*it);
      if (subid == SpecialConst::NullTrackId)
      {
        DissociatedTracks.push_back(*it);
      }
      else
      {
        *it_save = subid;
        ++it_save;
      }
    }
    nests.resize(it_save - it_begin);
  }

  // distribute, locate and levelup DissociatedTracks
  vector<HBTInt> ReceivedTracks;
  for (int root = 0; root < world.size(); root++)
  {
    HBTInt stacksize = DissociatedTracks.size();
    MPI_Bcast(&stacksize, 1, MPI_HBT_INT, root, world.Communicator);
    ReceivedTracks.resize(stacksize);
    MyBcast<HBTInt, vector<HBTInt>::iterator, vector<HBTInt>::iterator>(
      world, DissociatedTracks.begin(), ReceivedTracks.begin(), stacksize, MPI_HBT_INT, root);
    if (world.rank() != root)
    {
      for (auto &tid : ReceivedTracks)
      {
        auto subid = TrackHash.GetIndex(tid);
        if (subid != SpecialConst::NullTrackId) // located
          Subhalos[subid].Rank = 0;             // level up this DissociatedTrack
      }
    }
  }
}
void SubhaloSnapshot_t::GlobalizeTrackReferences()
/*translate subhalo references from index to trackIds*/
{
  int curr_snap = GetSnapshotIndex();
#pragma omp for
  for (HBTInt i = 0; i < Subhalos.size(); i++)
  {
    auto &subhalo = Subhalos[i];
    for (auto &&subid : subhalo.NestedSubhalos)
      subid = Subhalos[subid].TrackId;
  }
}

void SubhaloSnapshot_t::NestSubhalos(MpiWorker_t &world)
{
  LocalizeNestedIds(world);
  LevelUpDetachedSubhalos();
// collect detached(head) subhalos
#pragma omp single
  MemberTable.SubGroupsOfHeads.clear();
  MemberTable.SubGroupsOfHeads.resize(MemberTable.SubGroups.size());
#pragma omp parallel for
  for (HBTInt haloid = 0; haloid < MemberTable.SubGroups.size(); haloid++)
  {
    auto &subgroup = MemberTable.SubGroups[haloid];
    for (HBTInt i = 0; i < subgroup.size(); i++)
    {
      auto subid = subgroup[i];
      if (Subhalos[subid].Rank == 0)
        MemberTable.SubGroupsOfHeads[haloid].push_back(subid);
    }
  }
}

void SubhaloSnapshot_t::FillDepthRecursive(HBTInt subid, int depth)
{
  Subhalos[subid].Depth = depth;
  depth++;
  for (auto &&nestid : Subhalos[subid].NestedSubhalos)
  {
    FillDepthRecursive(nestid, depth);
  }
}

void SubhaloSnapshot_t::FillDepth()
{
#pragma omp for
  for (HBTInt grpid = 0; grpid < MemberTable.SubGroups.size(); grpid++)
    if (MemberTable.SubGroups[grpid].size())
      FillDepthRecursive(MemberTable.SubGroups[grpid][0], 0);
}

void SubhaloSnapshot_t::ExtendCentralNest()
{
#pragma omp for
  for (HBTInt haloid = 0; haloid < MemberTable.SubGroups.size(); haloid++)
  {
    auto &subgroup = MemberTable.SubGroups[haloid];
    auto &heads = MemberTable.SubGroupsOfHeads[haloid];
    if (subgroup.size() <= 1)
      continue;
    auto &central = Subhalos[subgroup[0]];
    if (central.Rank == 0) // already a central
    {
      central.NestedSubhalos.reserve(central.NestedSubhalos.size() + heads.size() - 1);
      for (auto &&h : heads)
        if (h != subgroup[0])
          central.NestedSubhalos.push_back(h);
    }
    else
    {
      central.Rank = 0; // promote to central
      for (auto &&h : heads)
        Subhalos[h].LevelUpDetachedMembers(Subhalos);
      central.NestedSubhalos.insert(central.NestedSubhalos.end(), heads.begin(), heads.end());
    }
  }
#pragma omp single
  MemberTable.SubGroupsOfHeads.clear();
}

void SubhaloSnapshot_t::LevelUpDetachedSubhalos()
/*
 * assign rank=0 to subhaloes that has drifted away from the hosthalo of its host-subhalo.
 */
{
  vector<char> IsHeadSub(Subhalos.size());
// record head list first, since the ranks are modified during LevelUpDetachedMembers().
#pragma omp parallel
  {
#pragma omp for
    for (HBTInt subid = 0; subid < Subhalos.size(); subid++)
      IsHeadSub[subid] = (Subhalos[subid].Rank == 0);

// promote centrals to detached
#pragma omp for
    for (HBTInt haloid = 0; haloid < MemberTable.SubGroups.size(); haloid++)
    {
      auto &subgroup = MemberTable.SubGroups[haloid];
      if (subgroup.size())
        Subhalos[subgroup[0]].Rank = 0;
    }
    {
      auto &subgroup = MemberTable.SubGroups[-1];
#pragma omp for
      for (HBTInt i = 0; i < subgroup.size(); i++) // break up all field subhalos
        Subhalos[subgroup[i]].Rank = 0;
    }
    // TODO: break up all orphans as well?

#pragma omp for
    for (HBTInt subid = 0; subid < Subhalos.size(); subid++)
      if (IsHeadSub[subid])
        Subhalos[subid].LevelUpDetachedMembers(Subhalos);
  }
}

void Subhalo_t::LevelUpDetachedMembers(vector<Subhalo_t> &Subhalos)
{
  HBTInt isave = 0;
  for (HBTInt i = 0; i < NestedSubhalos.size(); i++)
  {
    auto subid = NestedSubhalos[i];
    if (Subhalos[subid].HostHaloId != HostHaloId || Subhalos[subid].Rank == 0)
    {
      if (Subhalos[subid].Rank)
        Subhalos[subid].Rank = 0;
    }
    else
    {
      if (isave != i)
        NestedSubhalos[isave] = subid;
      isave++;
    }
    Subhalos[subid].LevelUpDetachedMembers(
      Subhalos); // recursively level up members. Note this can be further improved: if its members didn't follow it but
                 // stayed in the original host, then they should be added to the current NestedSubhalos, instead of
                 // being leveled up to rank 0. probably not necessary, since you did not actually check the host-sub
                 // but only adopted the historical relation.
  }
  NestedSubhalos.resize(isave); // remove detached ones from the list
}

class SubhaloMasker_t
{
public:
  unordered_set<HBTInt> ExclusionList;

  SubhaloMasker_t(HBTInt np_guess)
  {
    ExclusionList.reserve(np_guess);
  }
  /* This routine masks particles by giving preference to subhaloes deeper in
   * the hierarchy. */
  void Mask(HBTInt subid, vector<Subhalo_t> &Subhalos, int SnapshotIndex)
  {
    auto &subhalo = Subhalos[subid];
    for (auto nestedid :
         subhalo
           .NestedSubhalos) // TODO: do we have to do it recursively? satellites are already masked among themselves?
      Mask(nestedid, Subhalos, SnapshotIndex);

    if (subhalo.Nbound <= 1)
      return; // skip orphans

    if (subhalo.SnapshotIndexOfBirth == SnapshotIndex)
      return; // skip newly created centrals

    auto it_begin = subhalo.Particles.begin(), it_save = it_begin;
    for (auto it = it_begin; it != subhalo.Particles.end(); ++it)
    {
      auto insert_status = ExclusionList.insert(it->Id);
      if (insert_status.second) // inserted, meaning not excluded
      {
        if (it != it_save)
          *it_save = move(*it);
        ++it_save;
      }
    }
    subhalo.Particles.resize(it_save - it_begin);
  }

  HBTInt EstimateListSize(HBTInt subid, vector<Subhalo_t> &Subhalos,
                          const MappedIndexTable_t<HBTInt, HBTInt> &TrackHash)
  {

    HBTInt TotalSize = 0;
    auto &subhalo = Subhalos[subid];

    /* We go deeper in the hierarchy. Use TrackHash to navigate the Subhalo array. */
    for (auto nestedid : subhalo.NestedSubhalos)
      TotalSize += EstimateListSize(TrackHash.GetIndex(nestedid), Subhalos, TrackHash);

    TotalSize += subhalo.Particles.size();

    return TotalSize;
  }

  void CleanSource(HBTInt subid, vector<Subhalo_t> &Subhalos, const MappedIndexTable_t<HBTInt, HBTInt> &TrackHash)
  {
    /* Mask the 10 most bound tracer particles of every resolved subhalo in the
     * tree. We do this to not encounter issues during host finding. */
    MaskTopBottom(subid, Subhalos, TrackHash);

    /* Mask the remaining particles, such that they are preferentially given to
     * subhaloes deeper in the hierarchy. */
    MaskBottomTop(subid, Subhalos, TrackHash);
  }

  /* This routine masks particles by giving priority to subhaloes shallower
   * in the hierarchy. */
  void MaskTopBottom(HBTInt subid, vector<Subhalo_t> &Subhalos, const MappedIndexTable_t<HBTInt, HBTInt> &TrackHash)
  {
    /* We perform the masking first */
    auto &subhalo = Subhalos[subid];

    /* To keep track of how many tracers we have encountered so far. */
    HBTInt tracer_counter = 0;

    for (HBTInt i = 0; i < subhalo.Particles.size(); i++)
    {
      /* We have now ensured to keep the required number of most bound subset of
       * tracers for this subhalo. */
      if (tracer_counter == HBTConfig.MinNumTracerPartOfSub)
        break;

      /* Only mask tracers */
      if (subhalo.Particles[i].IsTracer())
      {
        /* We insert in the exclusion list so that more deeply nested subhaloes
         * do not remove this tracer when masking from bottom to top. */
        auto insert_status = ExclusionList.insert(subhalo.Particles[i].Id);

        /* We should have only just one bound unique tracer within the FOF level,
         * so we should not fail the insertion. */
        assert(insert_status.second == true);

        /* Since we do not change the size of subhalo particles at this stage,
         * we move forward the tracers, so we know how many to skip in the bottom
         * to top masking step. */
        std::swap(subhalo.Particles[i], subhalo.Particles[tracer_counter++]);
      }
    }

#ifndef NDEBUG
    /* Each resolved subhalo should contain at least this number of tracers
     * bound to it, hence we should have found a sufficient number. */
    if (subhalo.Nbound > 0)
    {
      assert(tracer_counter == HBTConfig.MinNumTracerPartOfSub);
      /* At this stage, we should have have all tracers at the MinNumTracerPartOfSub
       * most bound particles */

      for (int i = 0; i < HBTConfig.MinNumTracerPartOfSub; i++)
        assert(subhalo.Particles[i].IsTracer());
    }
#endif

    /* Update TracerIndex value */
    subhalo.SetTracerIndex(0);

    /* We go deeper in the hierarchy. Use TrackHash to navigate the Subhalo array. */
    for (auto nestedid : subhalo.NestedSubhalos)
      MaskTopBottom(TrackHash.GetIndex(nestedid), Subhalos, TrackHash);
  }

  /* This routine masks particles by giving priority to subhaloes deeper
   * in the hierarchy. */
  void MaskBottomTop(HBTInt subid, vector<Subhalo_t> &Subhalos, const MappedIndexTable_t<HBTInt, HBTInt> &TrackHash)
  {
    auto &subhalo = Subhalos[subid];

    /* We go deeper in the hierarchy. Use TrackHash to navigate the Subhalo array. */
    for (auto nestedid : subhalo.NestedSubhalos)
      MaskBottomTop(TrackHash.GetIndex(nestedid), Subhalos, TrackHash);

    /* Skip orphans */
    if (subhalo.Nbound <= 1)
      return;

#ifndef NDEBUG
    /* At this stage, we should have have all tracers at the MinNumTracerPartOfSub
     * most bound particles */
    for (int i = 0; i < HBTConfig.MinNumTracerPartOfSub; i++)
      assert(subhalo.Particles[i].IsTracer());
#endif

    /* At this point, we have navigated towards the bottom of the hierarchy.
     * Start masking bottom up. The iterators start after the last tracer we
     * shifted is. */
    HBTInt tracer_counter = HBTConfig.MinNumTracerPartOfSub;
    HBTInt NboundChange = 0;

    auto it_begin = subhalo.Particles.begin() + tracer_counter, it_save = it_begin;
    for (auto it = it_begin; it != subhalo.Particles.end(); ++it)
    {
      auto insert_status = ExclusionList.insert(it->Id);
      if (insert_status.second) // inserted, meaning not excluded
      {
        if (it != it_save)
          *it_save = move(*it);
        ++it_save;
      }
      /* Bound particle excluded; we will need to update Nbound. */
      else if ((it - subhalo.Particles.begin()) < subhalo.Nbound)
        NboundChange++;
    }

    /* Resize to achieve a clean source subhalo, i.e. no duplicate IDs across
     * subhaloes in the same FOF. This will prevent duplicates if the subhaloes
     * diverge in the next output. In this step we may remove enough particles
     * to make Nbound < MinNumPartOfSub, but the decision about its disruption is
     * made in the next output (e.g. it may reaccrete particles during unbinding) */
    subhalo.Particles.resize(it_save - subhalo.Particles.begin());

    /* We should not be checking whether the subhalo keeps all the particles, but rather
     * whether it retains at least MinNumTracerPartOfSub tracers */
    assert(subhalo.Particles.size() >= HBTConfig.MinNumTracerPartOfSub);

    /* This is being updated for consistency, but having an updated Nbound is
     * not a requirement within the code until after unbinding the next output. */
    subhalo.Nbound -= NboundChange;

#ifndef NDEBUG
    /* We should have retained at least 10 most bound tracers. */
    {
      int remaining_tracers = 0;
      for (int i = 0; i < subhalo.Nbound; i++)
      {
        remaining_tracers += subhalo.Particles[i].IsTracer() ? 1 : 0;
        if (remaining_tracers >= HBTConfig.MinNumTracerPartOfSub)
          break;
      }
      assert(remaining_tracers >= HBTConfig.MinNumTracerPartOfSub);
    }
#endif
  }
};

void SubhaloSnapshot_t::MaskSubhalos()
{
#pragma omp parallel for
  for (HBTInt i = 0; i < MemberTable.SubGroups.size(); i++)
  {
    auto &Group = MemberTable.SubGroups[i];
    if (Group.size() == 0)
      continue;
    auto &central = Subhalos[Group[0]];
    auto &nest = central.NestedSubhalos;
    auto old_membercount = nest.size();
    auto &heads = MemberTable.SubGroupsOfHeads[i];
    // update central member list (append other heads except itself)
    nest.insert(nest.end(), heads.begin() + 1, heads.end());
    SubhaloMasker_t Masker(central.Particles.size() * 1.2);
    Masker.Mask(Group[0], Subhalos, SnapshotIndex);
    nest.resize(old_membercount); // TODO: better way to do this? or do not change the nest for central?
  }
}

void SubhaloSnapshot_t::CleanTracks()
{
  /* Create correspondence between TrackId and index in Subhalo array */
  TrackKeyList_t Ids(*this);
  MappedIndexTable_t<HBTInt, HBTInt> TrackHash;
  TrackHash.Fill(Ids, SpecialConst::NullTrackId);

  /* Iterate over all subhalos in the present rank. */
#pragma omp parallel for
  for (HBTInt subid = 0; subid < Subhalos.size(); subid++)
  {
    /* Skip non-centrals and centrals with no subhalos */
    auto &central = Subhalos[subid];
    if ((central.Rank != 0) || (central.NestedSubhalos.size() == 0))
      continue;

    /* We need to use the TrackHash here, since the subids have been converted
     * to the global values. */
    SubhaloMasker_t Masker(central.Particles.size() * 1.2);

    // Try allocating sufficient memory for this to work
    HBTInt MaskerSize = Masker.EstimateListSize(subid, Subhalos, TrackHash);

    Masker.ExclusionList.reserve(MaskerSize * 1.2);
    Masker.CleanSource(subid, Subhalos, TrackHash);
  }
}

void SubhaloSnapshot_t::UpdateTracks(MpiWorker_t &world, const HaloSnapshot_t &halo_snap)
{
  /*renew ranks after unbinding*/
  RegisterNewTracks(world); // performance bottleneck here. no. just poor synchronization.

  // Update vmax for use in assigning rank within the host
#pragma omp parallel for if (ParallelizeHaloes)
  for (HBTInt i = 0; i < Subhalos.size(); i++)
    Subhalos[i].CalculateProfileProperties(*this);

#pragma omp parallel
  {
    MemberTable.SortMemberLists(Subhalos); // reorder, so the central might change if necessary
    ExtendCentralNest();
    MemberTable.AssignRanks(Subhalos);
    FillDepth();
#ifdef INCLUSIVE_MASS
    PurgeMostBoundParticles();
#endif
#pragma omp for
    for (HBTInt i = 0; i < Subhalos.size(); i++)
    {
      Subhalos[i].UpdateTrack(*this);
      HBTInt HostId = Subhalos[i].HostHaloId;
      if (HostId < 0)
        Subhalos[i].HostHaloId = -1;
      else
        Subhalos[i].HostHaloId = halo_snap.Halos[HostId].HaloId; // restore global haloid
    }
    GlobalizeTrackReferences();
    SetNestedParentIds();
  }
#pragma omp parallel for if (ParallelizeHaloes)
  for (HBTInt i = 0; i < Subhalos.size(); i++)
  {
#ifdef INCLUSIVE_MASS
    // Update Vmax etc using possibly updated particle list
    Subhalos[i].CalculateProfileProperties(*this);
#endif
    Subhalos[i].CalculateShape();

    for (int j = 0; j < 3; j++)
      Subhalos[i].ComovingAveragePosition[j] =
        position_modulus(Subhalos[i].ComovingAveragePosition[j], HBTConfig.BoxSize);
  }
}
