// TODO: unify the reference frame for specificProperties...
#include <algorithm>
#include <iostream>
#include <new>
#include <omp.h>

#include "datatypes.h"
#include "gravity_tree.h"
#include "snapshot_number.h"
#include "subhalo.h"

struct ParticleEnergy_t
{
  HBTInt pid;
  float E;
};
inline bool CompEnergy(const ParticleEnergy_t &a, const ParticleEnergy_t &b)
{
  return (a.E < b.E);
};
static HBTInt PartitionBindingEnergy(vector<ParticleEnergy_t> &Elist, const size_t len)
/*sort Elist to move unbound particles to the end*/
{ // similar to the C++ partition() func
  if (len == 0)
    return 0;
  if (len == 1)
    return Elist[0].E < 0;

  ParticleEnergy_t Etmp = Elist[0];
  auto iterforward = Elist.begin(), iterbackward = Elist.begin() + len;
  while (true)
  {
    // iterforward is a void now, can be filled
    while (true)
    {
      iterbackward--;
      if (iterbackward == iterforward)
      {
        *iterforward = Etmp;
        if (Etmp.E < 0)
          iterbackward++;
        return iterbackward - Elist.begin();
      }
      if (iterbackward->E < 0)
        break;
    }
    *iterforward = *iterbackward;
    // iterbackward is a void now, can be filled
    while (true)
    {
      iterforward++;
      if (iterforward == iterbackward)
      {
        *iterbackward = Etmp;
        if (Etmp.E < 0)
          iterbackward++;
        return iterbackward - Elist.begin();
      }
      if (iterforward->E > 0)
        break;
    }
    *iterbackward = *iterforward;
  }
}
static void PopMostBoundParticle(ParticleEnergy_t *Edata, const HBTInt Nbound)
{
  HBTInt imin = 0;
  for (HBTInt i = 1; i < Nbound; i++)
  {
    if (Edata[i].E < Edata[imin].E)
      imin = i;
  }
  if (imin != 0)
    swap(Edata[imin], Edata[0]);
}
class EnergySnapshot_t : public Snapshot_t
{
  HBTInt GetParticle(HBTInt i) const
  {
    return Elist[i].pid;
  }

public:
  ParticleEnergy_t *Elist;
  typedef vector<Particle_t> ParticleList_t;
  HBTInt N;
  const ParticleList_t &Particles;
  HBTReal MassFactor;
  EnergySnapshot_t(ParticleEnergy_t *e, HBTInt n, const ParticleList_t &particles, const Snapshot_t &epoch)
    : Elist(e), N(n), Particles(particles), MassFactor(1.)
  {
    Cosmology = epoch.Cosmology;
  };
  void SetMassUnit(HBTReal mass_unit)
  {
    MassFactor = mass_unit;
  }
  HBTInt size() const
  {
    return N;
  }
  HBTInt GetId(HBTInt i) const
  {
    return Particles[GetParticle(i)].Id;
  }
  HBTReal GetMass(HBTInt i) const
  {
    return Particles[GetParticle(i)].Mass * MassFactor;
  }
  const HBTxyz GetPhysicalVelocity(HBTInt i) const
  {
    return Particles[GetParticle(i)].GetPhysicalVelocity();
  }
  const HBTxyz &GetComovingPosition(HBTInt i) const
  {
    return Particles[GetParticle(i)].ComovingPosition;
  }
  double AverageVelocity(HBTxyz &CoV, HBTInt NumPart)
  /*mass weighted average velocity*/
  {
    HBTInt i, j;
    double svx, svy, svz, msum;

    if (0 == NumPart)
      return 0.;
    if (1 == NumPart)
    {
      copyHBTxyz(CoV, GetPhysicalVelocity(0));
      return GetMass(0);
    }

    svx = svy = svz = 0.;
    msum = 0.;
#pragma omp parallel for reduction(+ : msum, svx, svy, svz) if (NumPart > 100)
    for (i = 0; i < NumPart; i++)
    {
      HBTReal m = GetMass(i);
      const HBTxyz v = GetPhysicalVelocity(i);
      msum += m;
      svx += v[0] * m;
      svy += v[1] * m;
      svz += v[2] * m;
    }

    CoV[0] = svx / msum;
    CoV[1] = svy / msum;
    CoV[2] = svz / msum;
    return msum;
  }
  double AveragePosition(HBTxyz &CoM, HBTInt NumPart)
  /*mass weighted average position*/
  {
    HBTInt i, j;
    double sx, sy, sz, origin[3], msum;

    if (0 == NumPart)
      return 0.;
    if (1 == NumPart)
    {
      copyHBTxyz(CoM, GetComovingPosition(0));
      return GetMass(0);
    }

    if (HBTConfig.PeriodicBoundaryOn)
      for (j = 0; j < 3; j++)
        origin[j] = GetComovingPosition(0)[j];

    sx = sy = sz = 0.;
    msum = 0.;
#pragma omp parallel for reduction(+ : msum, sx, sy, sz) if (NumPart > 100)
    for (i = 0; i < NumPart; i++)
    {
      HBTReal m = GetMass(i);
      const HBTxyz &x = GetComovingPosition(i);
      msum += m;
      if (HBTConfig.PeriodicBoundaryOn)
      {
        sx += NEAREST(x[0] - origin[0]) * m;
        sy += NEAREST(x[1] - origin[1]) * m;
        sz += NEAREST(x[2] - origin[2]) * m;
      }
      else
      {
        sx += x[0] * m;
        sy += x[1] * m;
        sz += x[2] * m;
      }
    }
    sx /= msum;
    sy /= msum;
    sz /= msum;
    if (HBTConfig.PeriodicBoundaryOn)
    {
      sx += origin[0];
      sy += origin[1];
      sz += origin[2];
    }
    CoM[0] = sx;
    CoM[1] = sy;
    CoM[2] = sz;
    return msum;
  }
  void AverageKinematics(float &SpecificPotentialEnergy, float &SpecificKineticEnergy, float SpecificAngularMomentum[3],
                         HBTInt NumPart, const HBTxyz &refPos, const HBTxyz &refVel)
  /*obtain specific potential, kinetic energy, and angular momentum for the first NumPart particles
   * all quantities are physical

   * Note there is a slight inconsistency in the energy since they were calculated from the previous unbinding loop, but
   the refVel has been updated.
   */
  {
    if (NumPart <= 1)
    {
      SpecificPotentialEnergy = 0.;
      SpecificKineticEnergy = 0.;
      SpecificAngularMomentum[0] = SpecificAngularMomentum[1] = SpecificAngularMomentum[2] = 0.;
      return;
    }
    double E = 0., K = 0., AMx = 0., AMy = 0., AMz = 0., M = 0.;
#pragma omp parallel for reduction(+ : E, K, AMx, AMy, AMz, M) if (NumPart > 100)
    for (HBTInt i = 0; i < NumPart; i++)
    {
      HBTReal m = GetMass(i);
      E += Elist[i].E * m;
      const HBTxyz &x = GetComovingPosition(i);
      const HBTxyz v = GetPhysicalVelocity(i);
      double dx[3], dv[3];
      for (int j = 0; j < 3; j++)
      {
        dx[j] = x[j] - refPos[j];
        if (HBTConfig.PeriodicBoundaryOn)
          dx[j] = NEAREST(dx[j]);
        dx[j] *= Cosmology.ScaleFactor; // physical
        dv[j] = v[j] - refVel[j] + Cosmology.Hz * dx[j];
        K += dv[j] * dv[j] * m;
      }
      AMx += (dx[1] * dv[2] - dx[2] * dv[1]) * m;
      AMy += (dx[2] * dv[0] - dx[0] * dv[2]) * m;
      AMz += (dx[0] * dv[1] - dx[1] * dv[0]) * m;
      M += m;
    }
    E /= M;
    K *= 0.5 / M;
    SpecificPotentialEnergy = E - K;
    SpecificKineticEnergy = K;
    SpecificAngularMomentum[0] = AMx / M;
    SpecificAngularMomentum[1] = AMy / M;
    SpecificAngularMomentum[2] = AMz / M;
  }
};
inline void RefineBindingEnergyOrder(EnergySnapshot_t &ESnap, HBTInt Size, GravityTree_t &tree, HBTxyz &RefPos,
                                     HBTxyz &RefVel)
{ // reorder the first Size particles according to their self-binding energy
  auto &Elist = ESnap.Elist;
  auto &Particles = ESnap.Particles;
  tree.Build(ESnap, Size);
  vector<ParticleEnergy_t> Einner(Size);
#pragma omp parallel if (Size > 100)
  {
#pragma omp for
    for (HBTInt i = 0; i < Size; i++)
    {
      HBTInt pid = Elist[i].pid;
      Einner[i].pid = i;
      Einner[i].E = tree.BindingEnergy(Particles[pid].ComovingPosition, Particles[pid].GetPhysicalVelocity(), RefPos,
                                       RefVel, Particles[pid].Mass);
    }
#pragma omp single
    sort(Einner.begin(), Einner.end(), CompEnergy);
#pragma omp for
    for (HBTInt i = 0; i < Size; i++)
    {
      Einner[i] = Elist[Einner[i].pid];
    }
#pragma omp for
    for (HBTInt i = 0; i < Size; i++)
    {
      Elist[i] = Einner[i];
    }
  }
}
void Subhalo_t::Unbind(const Snapshot_t &epoch)
{ // the reference frame (pos and vel) should already be initialized before unbinding.

  /* We skip already existing orphans */
  if (!Particles.size())
  {
    Nbound = Particles.size();
    CountParticles();
    GetCorePhaseSpaceProperties();

    /* No bound particles, hence zero binding energies will be saved */
    if (HBTConfig.SaveBoundParticleBindingEnergies)
      ParticleBindingEnergies.clear();

    return;
  }

  /* We only expect (potentially) resolved subhaloes to make it here, or masked
   * out subhaloes (which should have at least one tracer particle if everything
   * is working correctly) */
  assert(Particles.size() >= 1);

  HBTInt MaxSampleSize = HBTConfig.MaxSampleSizeOfPotentialEstimate;
  bool RefineMostBoundParticle = (MaxSampleSize > 0 && HBTConfig.RefineMostBoundParticle);
  HBTReal BoundMassPrecision = HBTConfig.BoundMassPrecision;

  /* Need to initialise here, since orphans/disrupted objects do not call the
   * function used to set the value of TracerIndex (CountParticleTypes). This
   * prevents accessing entries beyond the corresponding particle array. */
  SetTracerIndex(0);

  /* Variables that store the centre of mass reference frame, which is used during
   * unbinding. In the first iteration, the centre of mass reference frame is that
   * of all particles associated to the subhalo in the previous output. Subsequent
   * iterations use the centre of mass frame of particles identified as bound. */
  HBTxyz OldRefPos, OldRefVel;
  auto &RefPos = ComovingAveragePosition;
  auto &RefVel = PhysicalAverageVelocity;

  GravityTree_t tree;
  tree.Reserve(Particles.size());
  Nbound = Particles.size(); // start from full set
  if (MaxSampleSize > 0 && Nbound > MaxSampleSize)
    random_shuffle(Particles.begin(), Particles.end()); // shuffle for easy resampling later.
  HBTInt Nlast;

  vector<ParticleEnergy_t> Elist(Nbound);
  for (HBTInt i = 0; i < Nbound; i++)
    Elist[i].pid = i;
  EnergySnapshot_t ESnap(Elist.data(), Elist.size(), Particles, epoch);
  bool CorrectionLoop = false;
  while (true)
  {
    if (CorrectionLoop)
    { // correct the potential due to removed particles
      HBTxyz RefVelDiff;
      epoch.RelativeVelocity(OldRefPos, OldRefVel, RefPos, RefVel, RefVelDiff);
      HBTReal dK = 0.5 * VecNorm(RefVelDiff);
      EnergySnapshot_t ESnapCorrection(&Elist[Nbound], Nlast - Nbound, Particles,
                                       epoch); // point to freshly removed particles
      tree.Build(ESnapCorrection);
#pragma omp parallel for if (Nlast > 100)
      for (HBTInt i = 0; i < Nbound; i++)
      {
        HBTInt pid = Elist[i].pid;
        auto &x = Particles[pid].ComovingPosition;
        auto v = Particles[pid].GetPhysicalVelocity();
        HBTxyz OldVel;
        epoch.RelativeVelocity(x, v, OldRefPos, OldRefVel, OldVel);
        Elist[i].E += VecDot(OldVel, RefVelDiff) + dK - tree.EvaluatePotential(x, 0);
      }
      Nlast = Nbound;
    }
    else
    {
      Nlast = Nbound;
      HBTInt np_tree = Nlast;
      if (MaxSampleSize > 0 && Nlast > MaxSampleSize) // downsample
      {
        np_tree = MaxSampleSize;
        ESnap.SetMassUnit((HBTReal)Nlast / MaxSampleSize);
      }
      tree.Build(ESnap, np_tree);
#pragma omp parallel for if (Nlast > 100)
      for (HBTInt i = 0; i < Nlast; i++)
      {
        HBTInt pid = Elist[i].pid;
        HBTReal mass;
        if (i < np_tree)
          mass = ESnap.GetMass(i); // to correct for self-gravity
        else
          mass = 0.; // not sampled in tree, no self gravity to correct
        Elist[i].E = tree.BindingEnergy(Particles[pid].ComovingPosition, Particles[pid].GetPhysicalVelocity(), RefPos,
                                        RefVel, mass);
#ifdef UNBIND_WITH_THERMAL_ENERGY
        Elist[i].E += Particles[pid].InternalEnergy;
#endif
      }
      ESnap.SetMassUnit(1.); // reset, no necessary
    }
    Nbound = PartitionBindingEnergy(Elist, Nlast); // TODO: parallelize this.
#ifdef NO_STRIPPING
    Nbound = Nlast;
#endif

    // Count the number of bound tracer particles
#ifdef DM_ONLY
    // All particles are tracers in DMO runs
    HBTInt Nbound_tracers = Nbound;
#else
    HBTInt Nbound_tracers = 0;
    for (HBTInt i = 0; i < Nbound; i += 1)
    {
      const auto &p = Particles[Elist[i].pid];
      if (p.IsTracer())
        Nbound_tracers += 1;
      if (Nbound_tracers >= HBTConfig.MinNumTracerPartOfSub)
        break; // We found enough, so no need to continue
    }
#endif

    /* Object has disrupted */
    if ((Nbound < HBTConfig.MinNumPartOfSub) || (Nbound_tracers < HBTConfig.MinNumTracerPartOfSub))
    {
      /* Store when it disrupted. */
      if (IsAlive())
        SnapshotIndexOfDeath = epoch.GetSnapshotIndex();

      /* The most bound positions of the new orphan were found when updating
       * every subhalo particles. Copy over to the comoving ones. For future
       * outputs, we will rely on UpdateMostBoundPosition instead */
      copyHBTxyz(ComovingAveragePosition, ComovingMostBoundPosition);
      copyHBTxyz(PhysicalAverageVelocity, PhysicalMostBoundVelocity);

      /* Do not allow the orphan to have any particles, so they can be subject to
       * unbinding in their parent. The particle array will be updated after this
       * subhalo has been done, when truncating the source. */
      Nbound = 0;
      Mbound = 0;

      break;
    }
    else
    {
      sort(Elist.begin() + Nbound, Elist.begin() + Nlast, CompEnergy); // only sort the unbound part
      HBTInt Ndiff = Nlast - Nbound;
      if (Ndiff < Nbound)
      {
        if (MaxSampleSize <= 0 || Ndiff < MaxSampleSize)
        {
          CorrectionLoop = true;
          copyHBTxyz(OldRefPos, RefPos);
          copyHBTxyz(OldRefVel, RefVel);
        }
      }

      /* The centre of mass frame is updated here */
      Mbound = ESnap.AverageVelocity(PhysicalAverageVelocity, Nbound);
      ESnap.AveragePosition(ComovingAveragePosition, Nbound);

      if (Nbound >= Nlast * BoundMassPrecision) // converge
      {
        if (!IsAlive())
          SnapshotIndexOfDeath = SpecialConst::NullSnapshotId; // clear death snapshot
        if (IsTrapped())
        {
          SnapshotIndexOfSink = SpecialConst::NullSnapshotId;
          SinkTrackId = SpecialConst::NullTrackId; // clear sinktrack as well
        }
        // update particle list
        sort(Elist.begin(), Elist.begin() + Nbound, CompEnergy); // sort the self-bound part

        /* We need to refine the most bound particle, as subsampling large subhaloes will lead to
         * incorrect ordering of binding energies. Hence, the most bound particle before this step
         * may not be the true most bound particle. */
        if (RefineMostBoundParticle && Nbound > MaxSampleSize)
        {
          /* If the number of bound particles is large, the number of particles used in this step scales with Nbound.
           * Using too few particles without this scaling would not result in a better centering. This is because it
           * would be limited to the (MaxSampleSize / Nbound) fraction of most bound particles, whose ranking can be
           * extremely sensitive to the randomness used during unbinding. */
          HBTInt SampleSizeCenterRefinement =
            max(MaxSampleSize, static_cast<HBTInt>(HBTConfig.BoundFractionCenterRefinement * Nbound));

          RefineBindingEnergyOrder(ESnap, SampleSizeCenterRefinement, tree, RefPos, RefVel);
        }

        // todo: optimize this with in-place permutation, to avoid mem alloc and copying.
        ParticleList_t p(Particles.size());
        for (HBTInt i = 0; i < Particles.size(); i++)
        {
          p[i] = Particles[Elist[i].pid];
          Elist[i].pid = i; // update particle index in Elist as well.
        }
        Particles.swap(p);

        /* Update the most bound coordinate. Note that for resolved subhaloes,
         * this is not necessarily a tracer particle. */
        copyHBTxyz(ComovingMostBoundPosition, Particles[0].ComovingPosition);
        copyHBTxyz(PhysicalMostBoundVelocity, Particles[0].GetPhysicalVelocity());
        break;
      }
    }
  }
  ESnap.AverageKinematics(SpecificSelfPotentialEnergy, SpecificSelfKineticEnergy, SpecificAngularMomentum, Nbound,
                          RefPos, RefVel); // only use CoM frame when unbinding and calculating Kinematics

  /* For orphans, this function call only sets it MboundType and NboundType equal to 0. For resolved objects, it
   * updates those fields, as well as the index of the most bound tracer particle.*/
  CountParticleTypes();

  /* At this stage we know the updated TracerIndex, so if we are bound we should
   * update the most bound ID. */
  if (IsAlive())
    MostBoundParticleId = Particles[GetTracerIndex()].Id;

  GetCorePhaseSpaceProperties();

  /* Store the binding energy information to save later */
  if (HBTConfig.SaveBoundParticleBindingEnergies)
  {
    ParticleBindingEnergies.resize(Nbound);
#pragma omp parallel for if (Nbound > 100)
    for (HBTInt i = 0; i < Nbound; i++)
      ParticleBindingEnergies[i] = Elist[i].E;
  }
}
void Subhalo_t::RecursiveUnbind(SubhaloList_t &Subhalos, const Snapshot_t &snap)
{
  /* Unbind all subhaloes that are nested deeper in the hierarchy of the current
   * one. */
  for (HBTInt i = 0; i < NestedSubhalos.size(); i++)
  {
    /* One of the children of the current subhalo */
    auto subid = NestedSubhalos[i];
    auto &subhalo = Subhalos[subid];
    subhalo.RecursiveUnbind(Subhalos, snap);

    /* The unbound particles of the child we just subjected to unbinding are
     * accreted to the source of the current subhalo. */
    Particles.insert(Particles.end(), subhalo.Particles.begin() + subhalo.Nbound, subhalo.Particles.end());
  }

  /* Unbind the current subhalo */
  Unbind(snap);

  /* Check if any of the subgroups deeper in this tree's hierarchy merge with it.
   * We update the particle list and the entries of the merged subhaloes if the
   * option is enabled. */
  bool HasExperiencedMerger = MergeRecursively(Subhalos, snap, *this);

  /* We need to subject the subhalo to unbinding once more, as it has accreted
   * new particles as a result of mergers. */
  if (HBTConfig.MergeTrappedSubhalos && HasExperiencedMerger)
    Unbind(snap);

  /* We are now sure about which particles are bound to this subhalo, so we can
   * safely pass the unbound ones to its parent and truncate the source.*/
}

void Subhalo_t::TruncateSource()
{
  HBTInt Nsource;
  if (Nbound <= 1)
    Nsource = Nbound;
  else
    Nsource = Nbound * HBTConfig.SourceSubRelaxFactor;
  if (Nsource > Particles.size())
    Nsource = Particles.size();
  Particles.resize(Nsource);
}

void SubhaloSnapshot_t::RefineParticles()
{ // it's more expensive to build an exclusive list. so do inclusive here.
  // TODO: ensure the inclusive unbinding is stable (contaminating particles from big subhaloes may hurdle the unbinding

#ifdef INCLUSIVE_MASS
#pragma omp parallel for schedule(dynamic, 1) if (ParallelizeHaloes)
  for (HBTInt subid = 0; subid < Subhalos.size(); subid++)
  {
    Subhalos[subid].Unbind(*this);
    Subhalos[subid].TruncateSource();
  }
#else
  HBTInt NumHalos = MemberTable.SubGroups.size();
#pragma omp parallel for schedule(dynamic, 1) if (ParallelizeHaloes)
  for (HBTInt haloid = 0; haloid < NumHalos; haloid++)
  {
    auto &subgroup = MemberTable.SubGroups[haloid];
    if (subgroup.size() == 0)
      continue;
    // add new satellites to central's NestedSubhalos
    auto &central = Subhalos[subgroup[0]];
    auto &nests = central.NestedSubhalos;
    auto old_membercount = nests.size();
    auto &heads = MemberTable.SubGroupsOfHeads[haloid];
    // update central member list (append other heads except itself)
    nests.insert(nests.end(), heads.begin() + 1, heads.end());
    central.RecursiveUnbind(Subhalos, *this);
    nests.resize(old_membercount); // restore old satellite list
  }
// unbind field subs
#pragma omp parallel
  {
    HBTInt NumField = MemberTable.SubGroups[-1].size();
#pragma omp for schedule(dynamic, 1) nowait
    for (HBTInt i = 0; i < NumField; i++)
    {
      HBTInt subid = MemberTable.SubGroups[-1][i];
      Subhalos[subid].Unbind(*this);
    }
    // unbind new-born subs
    HBTInt NumSubOld = MemberTable.AllMembers.size(), NumSub = Subhalos.size();
#pragma omp for schedule(dynamic, 1)
    for (HBTInt i = NumSubOld; i < NumSub; i++)
    {
      Subhalos[i].Unbind(*this);
    }
#pragma omp for schedule(dynamic, 1)
    for (HBTInt i = 0; i < NumSub; i++)
      Subhalos[i].TruncateSource();
  }
#endif
}
