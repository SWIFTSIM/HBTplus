/* if the particles are ordered in the snapshot and labelled from 1 to N in the group file, then define PID_ORDERED to avoid complex IndexTables*/
#include <iostream>
#include <new>
#include <algorithm>
#include <omp.h>
#include <cstdlib>

#include "hash.h"
//=====general ID2Index table======//
/* the hash-table implementation here is by sorting Id and use binsearch to locate keys*/
/* more general functions hcreate(),hsearch()... exists in glibc; but binsearch should be
 * more efficient than the general hsearch() I guess?*/

template <class Key_t, class Index_t>
inline bool CompPair(const IndexedKey_t<Key_t, Index_t> & a, const IndexedKey_t<Key_t, Index_t> & b)
{
  return (a.Key<b.Key);
};
template <class Key_t, class Index_t>
void MappedIndexTable_t<Key_t, Index_t>::Fill(const KeyList_t <Key_t, Index_t> &Keys)
{
  Index_t n=Keys.size();
	Map.resize(n);
	#pragma omp parallel for
	for(Index_t i=0;i<n;i++)
	{
		#ifdef PID_ORDERED
		Map[i].Key=i+1;
		#else
		Map[i].Key=Keys.GetKey(i);
		#endif
		Map[i].Index=i;
	}
	sort(Map.begin(), Map.end(), CompPair<Key_t, Index_t>);
}
template <class Key_t, class Index_t>
void MappedIndexTable_t<Key_t, Index_t>::Clear()
{
  vector <Pair_t>().swap(Map);
}
template <class Key_t, class Index_t>
inline int CompKeyWithPair(const void *a, const void *b)//used to sort Id in ascending order; 
{
  Key_t diff=(* static_cast<const Key_t *>(a))- static_cast<const IndexedKey_t<Key_t, Index_t> *> (b)->Key;
  if(diff>0) return 1;
  if(diff<0) return -1;
  return 0;
};
template <class Key_t, class Index_t>
Index_t MappedIndexTable_t<Key_t, Index_t>::GetIndex(const Key_t key) const
{//maybe implement the exception here? could be slow... test it first.
  if(key<0) return BaseClass_t::NullIndex;
  Pair_t *p=(Pair_t *) bsearch(&key,Map.data(),Map.size(),sizeof(Pair_t),CompKeyWithPair<Key_t, Index_t>);
  if(NULL==p) return BaseClass_t::NullIndex;  //no match
  return p->Index;  
}

template <class Key_t, class Index_t>
void FlatIndexTable_t<Key_t, Index_t>::Fill(const KeyList_t<Key_t, Index_t> &Keys)
{
  Index_t n=Keys.size();
	#ifndef PID_ORDERED
	KeyMax=Keys.GetKey(0);KeyMin=Keys.GetKey(0);
	for(Index_t i=1;i<n;i++)
	{
		if(Keys.GetKey(i)>KeyMax)
			KeyMax=Keys.GetKey(i);
		else if(Keys.GetKey(i)<KeyMin)	
			KeyMin=Keys.GetKey(i);
	}
	KeySpan=KeyMax-KeyMin+1;
	Index=new Index_t[KeySpan];
	Offset=KeyMin;
	Index-=Offset;/*shift PIndex by KeyMin element so that it is accessed through PIndex[KeyMin~KeyMax],
					i.e.,its subscript ranges the same as particle ID range. PID_all[PIndex[Id]]=Id;*/
	
	#pragma omp parallel 
	{
	#pragma omp for
	for(Key_t i=KeyMin;i<=KeyMax;i++)
		Index[i]=BaseClass_t::NullIndex; //initialize with -1, although not necessary
	/*====make ID index for query====*/
	#pragma omp for
	for(Index_t i=0;i<n;i++)
	  Index[Keys.GetKey(i)]=i;		
	}
	#else
	KeySpan=n;
	#endif
}
template <class Key_t, class Index_t>
void FlatIndexTable_t<Key_t, Index_t>::Clear()
{
	if(KeySpan)
	{
		KeySpan=0;
		#ifndef PID_ORDERED
		Index+=Offset;
		delete [] Index;
		#endif
	}
}
template <class Key_t, class Index_t>
Index_t FlatIndexTable_t<Key_t, Index_t>::GetIndex(const Key_t key) const
{
	#ifdef PID_ORDERED
	return key-1; //change from ID [1,N] to Index [0,N-1]
	#else
	if(key<KeyMin||key>KeyMax) return BaseClass_t::NullIndex;//no match
	return Index[key];
	#endif	
}