#!/usr/bin/env python

##################################################################################
'''Imports'''

import numpy as np
from typing import List, Tuple
import math
from shapely.geometry import Point, LineString
from shapely.ops import linemerge
import time
import multiprocessing as mp
import rasterio

from WINMOL_Analyzer import Stem
from WINMOL_Analyzer import Part
from WINMOL_Analyzer import Timer
from IO import get_bounds_from_profile

#System epsilon
epsilon = np.finfo(float).eps



##################################################################################
'''Vector operations'''

#######Parallel version of connect_stems

def connect_stems(stems:List[Stem], config) -> List[Stem]:
    
    max_distance=config.max_distance
    tolerance_angle=config.tolerance_angle
    min_length=config.min_length
    #Aggregate alligned stem segments to stems. Reconstructs occluded stems parts up die max_distance
    
    def return_callback(result):
        if result is not None:
            results.append(result)
    def error_callback(error):
        print(error, flush=True)
                
    t = Timer()
    t.start()
    pool = mp.Pool(mp.cpu_count()-1)
    print("#######################################################")     
    print("Reconstruction of windthrown stems")
    print("Connecting respective stem segments")
    cycle_nbr=1
    c_count=0
    out_count=0
    dublicates_count=0
    count_stem_parts=len(stems)
    global_change=True
    
    while global_change==True:
        #loop as long as stem parts can be attached to each other
        global_change=False     
        #sort by x coordinate
     #   stems.sort(key=lambda x: x.start.x)#, reverse=True)
        print("Cycle ", cycle_nbr)
        connected_stems=[]    
        
        stem_count=0
        can_count=0
        merged_count=0
        vote_count=0
        dub_count=0

        while stems:
            #loop while there are stems to extend
            
            #create a line of max 3 segments at both ends of the stem which represents the direction of the stem end
            if len(stems[0].path.coords)<4:
                lineStart = LineString([stems[0].path.coords[0], stems[0].path.coords[-1]])
                lineStop = LineString([stems[0].path.coords[0],stems[0].path.coords[-1]])             
            else:
                if len(stems[0].path.coords)<8:               
                    i=len(stems[0].path.coords)-2   
                else:
                    i=6
                lineStart = LineString([stems[0].path.coords[1], stems[0].path.coords[i]])
                lineStop = LineString([stems[0].path.coords[-(i+1)],stems[0].path.coords[-2]])  
            start_buffer=stems[0].start.buffer(max_distance, resolution=32)
            end_buffer=stems[0].stop.buffer(max_distance, resolution=32)
            
            #look at both ends for stems and search for potentionally canidates to append on stem[0]            
            stems_=[s for s in stems[1:] if start_buffer.contains(s.stop) or end_buffer.contains(s.start)]
            can_count=can_count+(len(stems_))
          
            canidates=[]
            votes=[]
            slaves=[]
            results =[]                
       
            r=[]    
            #Parallel computation of connectivity votes for stem parts potentionally appended to stems[0]
            for stem in stems_:
                r.append(pool.apply_async(calc_conectivity_votes, args=(stems[0],lineStart, lineStop,start_buffer, end_buffer, max_distance, tolerance_angle, min_length,  stem), callback=return_callback, error_callback=error_callback))             
   
            for r_ in r:
                r_.wait()     

            #prepare for evaluation
            change=[result[0] for result in results]
            votes = [result[1] for result in results]
            canidates = [result[2] for result in results]
            slaves = [result[3] for result in results]
            vote_counte=vote_count+len(votes)    
                    
            if any(change):                  
                index_min = min(range(len(votes)), key=votes.__getitem__)
                #stem is updated and the merged part is removed
                stems[0]=canidates[index_min] 
                stems.remove(slaves[index_min])
                #the new length is calculated
                stems[0].Length=stems[0].start.distance(stems[0].stop)
                global_change=True  
                c_count=c_count+1
                stems, dub_count_=remove_duplicates(stems,stems[0]) 
                dub_count=dub_count+dub_count_
                merged_count=merged_count+1

            else:
                #if no other stem part could be attached to stems[0], it is added to the export container and removed from the stem list
                connected_stems.append(stems[0])              
                stems.remove(stems[0])
                stem_count=stem_count+1
                dublicates_count=dublicates_count+dub_count
                
                
                can_count=0
                merged_count=0
                dub_count=0
        
        #all stems have veen observed and passed to connected stems. s
        stems=connected_stems
        cycle_nbr=cycle_nbr+1

    pool.close()
    
    connected_stems=[]    
    for stem in stems:
        if stem.Length > min_length:
            connected_stems.append(stem)
        else:
            out_count=out_count+1
    connected_stems, dub_count_2=remove_duplicates(connected_stems)
    dub_count_=dublicates_count+dub_count_2
    print("")
    print(count_stem_parts,"stem segments analyzed")   
    print(c_count,"stem segments appended to other stems")
    print(dublicates_count, "duplicates are removed")
    print(out_count, "stem fragments with a length less than ", min_length,"m are filtered out")
    print("final number of stems",  len(connected_stems))
    t.stop()
    print("#######################################################")   
    print("")
    return connected_stems                                                                            

def calc_conectivity_votes(stems0:Stem, lineStart:LineString, lineStop:LineString,start_buffer, end_buffer, max_distance, tolerance_angle, min_length, stem:Stem) -> (bool, List[float], List[Stem], List[Stem]):
    #Calculate votes for the aggregation of stem parts to stems
    if stem == stems0:
        if stem.start==stems0.start:
            if stem.stop==stems0.stop:
                print("Alerta!!!!", flush=True)
                print("stem0: ", list(stems0.path.coords), flush=True)
                print("stem: ", list(stem.path.coords), flush=True)      
    change=False
    votes=[]
    canidates=[]
    slaves=[]
    
    if len(stem.path.coords)<4:
        e_lineStart = LineString([stem.path.coords[0], stem.path.coords[-1]])
        e_lineStop = LineString([stem.path.coords[0], stem.path.coords[-1]])  
    else:
        if len(stem.path.coords)<8:
            k=len(stem.path.coords)-2   
        else:
            k=6
        e_lineStart = LineString([stem.path.coords[1], stem.path.coords[k]])
        e_lineStop = LineString([stem.path.coords[-(k+1)], stem.path.coords[-2]])     
                    
    ang_lSp_elSt=abs(ang(lineStop.coords, e_lineStart.coords))                 
    ang_elSp_lSt=abs(ang(e_lineStop.coords,lineStart.coords))

    if (end_buffer.contains(stem.start)and ang_lSp_elSt<tolerance_angle):       
        missing_part=LineString([stems0.stop, stem.start])
        missing_part_=LineString([stems0.path.coords[-2], stem.path.coords[1]])
        dist_f=1-(1/(3+max_distance-stems0.stop.distance(stem.start))**0.5)
        ang_lSp_mp=abs(ang(lineStop.coords, missing_part_.coords))
        ang_mp_elSt=abs(ang(missing_part_.coords, e_lineStart.coords))
                    
        if (ang_lSp_elSt<(tolerance_angle*dist_f) and ang_lSp_mp <(tolerance_angle*dist_f) and ang_mp_elSt<(tolerance_angle*dist_f) and stems0.start.distance(stem.stop)<35):  

            if (len(stems0.path.coords)>2 and len(stem.path.coords)>2): 
                start=LineString(stems0.path.coords[:-1])
                end=LineString(stem.path.coords[1:])
                new_path=linemerge([start,missing_part_,end])            
            else:
                if (len(stems0.path.coords)>2  and len(stem.path.coords)==2):
                    start=LineString(stems0.path.coords[:-1])
                    new_path=linemerge([start,missing_part_])                  
                else:
                    if (len(stems0.path.coords)==2  and len(stem.path.coords)>2):
                        end=LineString(stem.path.coords[1:])
                        new_path=linemerge([missing_part_,end])                       
                    else:
                        if (len(stems0.path.coords)==2  and len(stem.path.coords)==2):
                            new_path=missing_part_

            change=True 
            canidate=stems0
            canidate.path=new_path
            canidate.stop=stem.stop
            slave=stem
            vote=((1+ang_lSp_elSt+ang_lSp_mp + ang_mp_elSt)/tolerance_angle)*canidate.start.distance(canidate.stop)**2+stems0.stop.distance(stem.start)**2*(1+ang_lSp_elSt+ang_lSp_mp + ang_mp_elSt)/tolerance_angle
            canidates.append(canidate)
            votes.append(vote)
            slaves.append(slave)

    if (start_buffer.contains(stem.stop)and ang_elSp_lSt<tolerance_angle):
            missing_part=LineString([stem.stop, stems0.start])
            missing_part_=LineString([stem.path.coords[-2], stems0.path.coords[1]])
            dist_f=1-(1/(3+max_distance-stem.stop.distance(stems0.start))**0.5)
            ang_elSp_mp=abs(ang(e_lineStop.coords, missing_part_.coords))
            ang_mp_lSt=abs(ang(missing_part_.coords, lineStart.coords))

            if  (ang_elSp_lSt<(tolerance_angle*dist_f) and ang_elSp_mp<(tolerance_angle*dist_f) and abs(ang(missing_part_.coords, lineStart.coords))<(tolerance_angle*dist_f) and stem.start.distance(stems0.stop)<35): 
                if (len(stem.path.coords)>2 and len(stems0.path.coords)>2):
                    start=LineString(stem.path.coords[:-1])
                    end=LineString(stems0.path.coords[1:])
                    new_path=linemerge([start,missing_part_,end])
                else:
                    if (len(stem.path.coords)>2  and len(stems0.path.coords)==2):
                        start=LineString(stem.path.coords[:-1])
                        new_path=linemerge([start,missing_part_])
                    else:
                        if (len(stem.path.coords)==2  and len(stems0.path.coords)>2):
                            end=LineString(stems0.path.coords[1:])
                            new_path=linemerge([missing_part_,end])
                        else:
                            if (len(stem.path.coords)==2  and len(stems0.path.coords)==2):
                                new_path=missing_part_

                change=True
                canidate=stems0
                canidate.path=new_path
                canidate.start=stem.start
                slave=stem
                vote=((1+ang_elSp_lSt+ang_elSp_mp + ang_mp_lSt)/tolerance_angle)*canidate.start.distance(canidate.stop)**2+stem.stop.distance(stems0.start)**2*(1+ang_elSp_lSt+ang_elSp_mp + ang_mp_lSt)/tolerance_angle
                canidates.append(canidate)
                votes.append(vote)
                slaves.append(slave)     
                
    if change == True:
        index_min = min(range(len(votes)), key=votes.__getitem__)     
        return (True, votes[index_min], canidates[index_min], slaves[index_min]) 
    else:
        return (False, math.inf, None, None)
    
#####Helper functions vector operations    
    

def build_stem_parts(segments:List[Part]):
    #######Converts the List of [Part] containing Tuples[int] into List of [Stem] consisiting of shapely geometries
    t = Timer()
    t.start()
    print("#######################################################") 
    print("Build stem segments")
    stems=[]
    for i in range(len(segments)):
            if segments[i].start[1]>=segments[i].stop[1]:
                h=segments[i].start
                segments[i].start=segments[i].stop
                segments[i].stop=h
                segments[i].path.reverse()
            else:
                h=segments[i].start
                segments[i].start=segments[i].stop
                segments[i].stop=h
                segments[i].path.reverse()
    segments=set(segments)
    for seg in segments:
            l=math.dist(seg.start, seg.stop)
            stem=Stem(Point(seg.start),Point(seg.stop),LineString(seg.path),[],[],[],[],l,None)
            stems.append(stem)
   
    print(len(stems), "stems segments build")

    t.stop()
    print("#######################################################")   
    print("")     
    return stems

def rebuild_endnodes_from_stems(stems: List[Stem])->List[Point]:
    t = Timer()
    t.start()
    print("#######################################################")   
    print("Rebuild endnodes from stems")
    nodes = []
    for s in stems:
        nodes.append(s.start.coords)
        nodes.append(s.stop.coords)   
    t.stop()    
    print("#######################################################")
    print("")
    return nodes

def remove_duplicates(stems:List[Stem], stems0=None) -> List[Stem]:  
    #Removes duplicates from stem list
    stems.sort(key=lambda x: x.Length, reverse=True)
    stems_=[]
    count=0
    if type(stems0) is Stem:
        for s in stems:
            if stems0.path.buffer(0.3).contains(s.path):
                stems.remove(s)
                count=count+1
        stems_=stems
        stems_.append(stems0)
    else:
        while stems:
            for s in stems[1:]:
                if stems[0].path.buffer(0.3).contains(s.path):
                    stems.remove(s)
                    count=count+1
            stems_.append(stems[0])
            stems.remove(stems[0])
    return stems_, count

def restore_geoinformation(stems: List[Stem], config, profile):
    #remove padding and restore geoinformation of the stems
    t = Timer()
    t.start()
    
    print("#######################################################")   
    print("Restoring geoinformation")      
   
    px_size=profile['transform'][0]
    bounds=get_bounds_from_profile(profile)
    padding=int(config.max_tree_height/px_size)+1    
    
    for j in range(len(stems)):    
        stems[j].start=(bounds.left+(stems[j].start[1]-padding)*px_size, bounds.top-(stems[j].start[0]-padding)*px_size)
        stems[j].stop=(bounds.left+(stems[j].stop[1]-padding)*px_size, bounds.top-(stems[j].stop[0]-padding)*px_size)
        for k in range(len(stems[j].path)):
            stems[j].path[k]=(bounds.left+(stems[j].path[k][1]-padding)*px_size, bounds.top-(stems[j].path[k][0]-padding)*px_size)
        stems[j].Length=math.dist(stems[j].start,stems[j].stop)
    t.stop()    
    print("#######################################################")  
    print("") 
    return  stems    

#######Helper functions for skeleton operations   

def create_vector(line):
    #Creates a normalized vecor from LineStrings or Tulple[Tuple[int]] 
    if type(line)=='LineString':
        v= [line.coords[-1][0]-line.coords[0][0], line.coords[-1][1]-line.coords[0][1]]
    else:
        v= [(line[1][0]-line[0][0]), (line[1][1]-line[0][1])]   
    return v/(np.linalg.norm(v)+epsilon)

                     
def ang(lineA, lineB):
    #Calculates the angle between 2 vectors
    vA = create_vector(lineA)
    vB = create_vector(lineB)
  #  dot_product = np.dot(vA/(np.linalg.norm(vA)+epsilon), vB/(np.linalg.norm(vB)+epsilon))
    dot_product = np.dot(vA, vB)
    dot_product=np.clip(dot_product, -1, 1)
    angle = np.arccos(dot_product)
    ang_deg = np.degrees(angle)%360
    if ang_deg > 180:
        ang_deg = ang_deg-360
    return ang_deg


