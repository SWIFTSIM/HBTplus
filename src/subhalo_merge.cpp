#include <algorithm>
#include <iostream>
#include <new>
#include <omp.h>
#include <unordered_set>

#include "datatypes.h"
#include "snapshot_number.h"
#include "subhalo.h"

#define NumPartCoreMax 20
#define PhaseSpaceDistanceThreshold 2.

/* Computes distance in phase space between the current subhalo and a reference
 * one. */
float Subhalo_t::PhaseSpaceDistance(const Subhalo_t &ReferenceSubhalo)
{
  float position_offset = PeriodicDistance(ReferenceSubhalo.CoreComovingPosition, CoreComovingPosition);
  float velocity_offset = Distance(ReferenceSubhalo.CorePhysicalVelocity, CorePhysicalVelocity);
  return position_offset / ReferenceSubhalo.CoreComovingSigmaR + velocity_offset / ReferenceSubhalo.CorePhysicalSigmaV;
}

/* Check if the current subhalo satisfies merger criterion with a reference one. */
bool Subhalo_t::AreOverlappingInPhaseSpace(const Subhalo_t &ReferenceSubhalo)
{
  return PhaseSpaceDistance(ReferenceSubhalo) < PhaseSpaceDistanceThreshold; 
}

/* Store information about the merger that has just occured. */
void Subhalo_t::SetMergerInformation(const int &ReferenceTrackId, const int &CurrentSnapshotIndex)
{
  /* When this occured */
  SnapshotIndexOfSink = CurrentSnapshotIndex;
  
  /* Which TrackId it merged with */
  SinkTrackId = ReferenceTrackId;

  /* Death time if this merger caused it. */
  if(IsAlive())
    SnapshotIndexOfDeath = CurrentSnapshotIndex;
}

/* Recursively checks in a depth-first approach whether any of the subhaloes 
 * contained within the hierarchical subtree of ReferenceSubhalo overlap in phase-
 * space with it. If any resolved subhalo is found to be overlapping, we will
 * unbind the ReferenceSubhalo once more. */
bool Subhalo_t::MergeRecursively(SubhaloList_t &Subhalos, const Snapshot_t &snap, Subhalo_t &ReferenceSubhalo)
{
  bool ExperiencedMerger = false;

  /* Do not consider unbound subhaloes as eligible to accrete particles through 
   * a merger*/
  if(ReferenceSubhalo.Nbound <= 1)
    return ExperiencedMerger;

  /* Iterate over all the subhaloes who share this subhalo in its hierarchy 
   * tree. */
  for (HBTInt i = 0; i < NestedSubhalos.size(); i++)
  {
    /* One of the children of the current subhalo */
    auto ChildIndex = NestedSubhalos[i];
    auto &ChildSubhalo = Subhalos[ChildIndex];

    /* Go further down the hierarchy if possible */
    ExperiencedMerger = ChildSubhalo.MergeRecursively(Subhalos, snap, ReferenceSubhalo);
  }

  /* Only deal with present subhalo if is not already trapped and it is not the 
   * reference subhalo. */
  if(!IsTrapped() && (this != &ReferenceSubhalo))
  {
    if(AreOverlappingInPhaseSpace(ReferenceSubhalo))
    {
      SetMergerInformation(ReferenceSubhalo.TrackId, snap.GetSnapshotIndex());

      /* Flag if the subhalo was previously resolved, and hence the reference
       * subhalo will accrete particles resulting from the merger. */
      ExperiencedMerger = Nbound > 1;

      /* If enabled, pass the particles to the reference subhalo we merged to. */
      if(HBTConfig.MergeTrappedSubhalos)
        MergeTo(ReferenceSubhalo);
    }    
  }

  return ExperiencedMerger;
}

/* Computes the mass-weighted position and velocity of a subset of the most bound
 * particles, as well as the 1D dispersion of each quantity. We first try using
 * the tracer particles, but if they are insufficient in number, we use all types.*/
void Subhalo_t::GetCorePhaseSpaceProperties()
{
  /* Need to handle orphans differently, since they have no particles
   * associated to them explicitly*/
  if (Nbound == 0)
  {
    copyHBTxyz(CoreComovingPosition, ComovingMostBoundPosition);
    copyHBTxyz(CorePhysicalVelocity, PhysicalMostBoundVelocity);
    CoreComovingSigmaR = 0.;
    CorePhysicalSigmaV = 0.;
    return;
  }

  /* Initalize variables used to accumulate position, velocity, mass etc */
  HBTInt NumPart = 0;
  double msum = 0; 
  vector<double> pos(3,0), pos2(3,0);
  vector<double> vel(3,0), vel2(3,0);

  // Use first particle as reference point for box wrap
  vector<double> origin(3,0);
  if (HBTConfig.PeriodicBoundaryOn)
    for (int j = 0; j < 3; j++)
      origin[j] = Particles[0].ComovingPosition[j];

  // Might need to make two passes through the particles
  for (int pass_nr = 0; pass_nr < 2; pass_nr += 1)
  {
    // Loop over particles in the subhalo
    for (int i = 0; i < Nbound; i++)
    {
      const int is_tracer = Particles[i].IsTracer();

      // First pass uses tracers only, whereas second pass uses non-tracers only
      if ((is_tracer && (pass_nr == 0)) || (!is_tracer && (pass_nr == 1)))
      {
        NumPart += 1;
        HBTReal m = Particles[i].Mass;
        msum += m;

        for(int dim = 0; dim < 3; dim++)
        {
          /* Handle position and PCB, if required */
          double dx = Particles[i].ComovingPosition[dim];
          if (HBTConfig.PeriodicBoundaryOn)
            dx = NEAREST(dx - origin[dim]);

          pos[dim] += m * dx;
          pos2[dim] += m * dx * dx;
          
          /* Handle velocity */
          double dv = Particles[i].PhysicalVelocity[dim];
          vel[dim] += m * dv;
          vel2[dim] += m * dv * dv;
        }
      }
      if (NumPart == NumPartCoreMax)
        break;
      /* Next particle in subhalo */
    }
    if (NumPart == NumPartCoreMax)
      break;
    /* Next pass */
  }

  for(int dim = 0; dim < 3; dim++)
  {
    pos[dim] /= msum;
    pos2[dim] /= msum;
    CoreComovingPosition[dim] = pos[dim];
    
    if (HBTConfig.PeriodicBoundaryOn)
      CoreComovingPosition[dim] += origin[dim];

    pos2[dim] -= pos[dim] * pos[dim];
    
    vel[dim]  /= msum;
    vel2[dim]  /= msum;
    CorePhysicalVelocity[dim] = vel[dim];
    vel2[dim] -= vel[dim] * vel[dim];
  }

  CoreComovingSigmaR = sqrt(pos2[0] + pos2[1] + pos2[2]); 
  CorePhysicalSigmaV = sqrt(vel2[0] + vel2[1] + vel2[2]);
}

void Subhalo_t::MergeTo(Subhalo_t &host)
{
  if (Nbound <= 1)
    return; // skip orphans and nulls

#ifndef INCLUSIVE_MASS
  HBTInt np_max = host.Particles.size() + Particles.size();
  unordered_set<HBTInt> UniqueIds(np_max);
  for (auto &&p : host.Particles)
    UniqueIds.insert(p.Id);
  host.Particles.reserve(np_max);
  for (auto &&p : Particles)
    if (UniqueIds.insert(p.Id).second) // inserted, meaning not excluded
      host.Particles.push_back(p);
  host.Nbound += Nbound;
#endif

  /* We will copy the information required to save the orphan in this output.
   * For future outputs, we will rely on UpdateMostBoundPosition instead. We need
   * to do it here, since the MostBoundPosition and MostBoundVelocity of tracks
   * are based on the ACTUAL MOST BOUND PARTICLE. Orphans are based on the
   * MOST BOUND TRACER PARTICLE. */
  Mbound = Particles[GetTracerIndex()].Mass;
  MostBoundParticleId = Particles[GetTracerIndex()].Id;
  copyHBTxyz(ComovingMostBoundPosition, Particles[GetTracerIndex()].ComovingPosition);
  copyHBTxyz(PhysicalMostBoundVelocity, Particles[GetTracerIndex()].GetPhysicalVelocity());
  copyHBTxyz(ComovingAveragePosition, ComovingMostBoundPosition);
  copyHBTxyz(PhysicalAverageVelocity, PhysicalMostBoundVelocity);

  /* Do not allow the orphan to have any particles, so they can be subject to
   * unbinding in their parent. The particle array will be updated after this
   * subhalo has been done. */
  Nbound = 0;
  Particles.resize(0);
  CountParticles();
}
