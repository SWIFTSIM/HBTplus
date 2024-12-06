using namespace std;
#include <cstdlib>
#include <iostream>
#include <string>
#include <omp.h>

#include "src/mpi_wrapper.h"
#include "src/datatypes.h"
#include "src/config_parser.h"
#include "src/snapshot.h"
#include "src/halo.h"
#include "src/subhalo.h"
#include "src/mymath.h"
#include "src/particle_exchanger.h"

int main(int argc, char **argv)
{
 MPI_Init(&argc, &argv);
 MpiWorker_t world(MPI_COMM_WORLD);
#ifdef _OPENMP
 //omp_set_nested(0);
 omp_set_max_active_levels(1); //max_active_level 0: no para; 1: single layer; >1: nest enabled
#endif
   
  int snapshot_start, snapshot_end;
  if(0==world.rank())
  {
	ParseHBTParams(argc, argv, HBTConfig, snapshot_start, snapshot_end);
	mkdir(HBTConfig.SubhaloPath.c_str(), 0755);
	HBTConfig.DumpParameters();
    
    cout<<argv[0]<<" run using "<<world.size()<<" mpi tasks";
    #ifdef _OPENMP
    #pragma omp parallel
    #pragma omp master
    cout<<", each with "<<omp_get_num_threads()<<" threads";
    #endif
    cout<<endl;
  }
  HBTConfig.BroadCast(world, 0, snapshot_start, snapshot_end);

  SubhaloSnapshot_t subsnap;
  
  subsnap.Load(world, snapshot_start-1, SubReaderDepth_t::SrcParticles);
  
  Timer_t global_timer;
  ofstream time_log;
  if(world.rank()==0)
  {
  time_log.open(HBTConfig.SubhaloPath+"/timing.log", fstream::out|fstream::app);
  time_log<<fixed<<setprecision(1);//<<setw(8);
  }
  for(int isnap=snapshot_start;isnap<=snapshot_end;isnap++)
  {
	global_timer.Tick("start", world.Communicator);
	ParticleSnapshot_t partsnap;
	partsnap.Load(world, isnap);
	global_timer.Tick("read_snap", world.Communicator);
	subsnap.SetSnapshotIndex(isnap);
	HaloSnapshot_t halosnap;
	halosnap.Load(world, isnap);
	
	global_timer.Tick("read_halo",world.Communicator);
// 	cout<<"updating halo particles...\n";
	halosnap.UpdateParticles(world, partsnap);
	global_timer.Tick("update_halo",world.Communicator);
// 	if(world.rank()==0) cout<<"updateing subsnap particles...\n";
	subsnap.UpdateParticles(world, partsnap);
	
	global_timer.Tick("update_subhalo",world.Communicator);
	subsnap.AssignHosts(world, halosnap, partsnap);
	global_timer.Tick("assign_hosts",world.Communicator);
	subsnap.PrepareCentrals(world, halosnap);
	
	global_timer.Tick("prepare_centrals",world.Communicator);
	if(world.rank()==0) cout<<"unbinding...\n";
	subsnap.RefineParticles();
	
	global_timer.Tick("unbind",world.Communicator);
	subsnap.MergeSubhalos();
	
	global_timer.Tick("merge_subhalos",world.Communicator);
	subsnap.UpdateTracks(world, halosnap);
	
	global_timer.Tick("update_tracks",world.Communicator);
	subsnap.Save(world);
	
	global_timer.Tick("write_subhalos",world.Communicator);
	if(world.rank()==0)
	{
	time_log<<isnap;
	for(int i=1;i<global_timer.Size();i++)
	  time_log<<"\t"<<global_timer.GetSeconds(i);
	time_log<<endl;
	}
	global_timer.Reset();
  }
  
  MPI_Finalize();
  return 0;
}
