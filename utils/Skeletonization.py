#!/usr/bin/env python

##################################################################################
'''Imports'''

import numpy as np

from typing import List, Tuple
import scipy.ndimage.measurements
from skimage import morphology#, segmentation
import math
import multiprocessing as mp

from stand_alone.WINMOL_Analyzer import Part
from stand_alone.WINMOL_Analyzer import Timer

#System epsilon
epsilon = np.finfo(float).eps

               
################################################################################## 
'''Skeleton operations'''

#####Find Nodes and connected segments in the Skeleton

def find_segments(pred, config, profile) -> (List[Part],List[Tuple[int]]):
    t = Timer()
    t.start()
       
    print("#######################################################")   
    print("Skeletonize Image") 
    
    px_size=profile['transform'][0]
    min_length=config.min_length/4
    padding=int(config.max_tree_height/px_size)+1
    pred=np.pad(pred,((padding,padding),(padding,padding)),'constant', constant_values=False)
    
    # binarize image        
    pred[np.where( pred < 0.5 )] = 0
    pred[np.where( pred > 0.5 )] = 1
    
    skel = morphology.skeletonize(pred)
        
    t.stop()
    print("#######################################################")  
    print("")  

    endnodes, skel = get_nodes(skel,padding)
    segments, skel = find_skeleton_segments(skel, endnodes, math.floor(min_length/px_size),padding)
    segments = refine_skeleton_segments(segments,skel, math.floor(min_length/px_size))
                         
    return segments

####get nodes

def get_nodes(skel: np.ndarray, padding:int) -> List[Tuple[int, int]]:    
    t = Timer()
    t.start()
    print("#######################################################") 
    print("Splitting the skeleton into segments and detecting endnodes") 
    
    skel, dn_count=remove_dense_skeleton_nodes(skel)   

    print("Dense nodes removed: ", dn_count) 

    endnodes, branchpoints = find_skeleton_nodes(skel)
    bp_count=len(branchpoints)
    while len(branchpoints)>0: 
            skel=remove_branchpoints_from_skel(skel, endnodes, branchpoints)   
            endnodes, branchpoints = find_skeleton_nodes(skel)  
            bp_count=bp_count+len(branchpoints)
    skel = morphology.skeletonize(skel)  
    print("Brachpoints removed: ", bp_count)        
    print("Detected endnodes: ", len(endnodes))
    t.stop()
    print("#######################################################") 
    print("")
    return endnodes, skel

def remove_dense_skeleton_nodes(skel: np.ndarray) -> List[Tuple[int, int]]:
    """Remove "dense" (2x2 or larger) regions in the skeleton.
    """
    dense_nodes = morphology.binary_erosion(np.pad(skel, 1), np.ones((2, 2)))[1:-1, 1:-1]
    labeled_array, num_features = scipy.ndimage.measurements.label(dense_nodes)
    centers = scipy.ndimage.measurements.center_of_mass(dense_nodes, labeled_array, [*range(1, num_features+1)])
    count=len(centers)

    skel[np.where(dense_nodes==True)]=False

    return skel, count


def find_skeleton_nodes(skel: np.ndarray) -> List[Tuple[int]]:
    """Find nodes in a skeletonized bitmap.
    """
    skel = np.pad(skel, 1)
    item = skel.item
    endnodes =[]
    branchpoints = []
    width, height = skel.shape
    for x in range(1, width-1 ):
        for y in range(1, height-1):
            if item(x, y) != 0 and is_endpoint_or_branchpoint(x, y, skel)=="branchpoint":
                branchpoints.append((x-1 , y-1 )) # (-1, -1) removes the padding
            if item(x, y) != 0 and is_endpoint_or_branchpoint(x, y, skel)=="endpoint":
                endnodes.append((x-1 , y-1 )) # (-1, -1) removes the padding
    return endnodes, branchpoints

def is_endpoint_or_branchpoint(x, y, skel):
    """Checks the number of neighbours belonging to the skeleton around around a pixel. 
    If a point has 1 neighbour, it is considered as an endnode, if a point has 3 or more neigbours, it is considered as branchpoint.
    """
    item = skel.item
    p2 = item(x - 1, y)
    p3 = item(x - 1, y + 1)
    p4 = item(x, y + 1)
    p5 = item(x + 1, y + 1)
    p6 = item(x + 1, y)
    p7 = item(x + 1, y - 1)
    p8 = item(x, y - 1)
    p9 = item(x - 1, y - 1)

    # The function A(p1),
    # where p1 is the pixel whose neighborhood is beeing checked
    components = (        
        (p2 == 0 and p3 == 1)
        + (p3 == 0 and p4 == 1)
        + (p4 == 0 and p5 == 1)
        + (p5 == 0 and p6 == 1)
        + (p6 == 0 and p7 == 1)
        + (p7 == 0 and p8 == 1)
        + (p8 == 0 and p9 == 1)
        + (p9 == 0 and p2 == 1)
        )
    if (components >= 3):
        return "branchpoint"
    if (components == 1):
        return "endpoint"
    return False


def remove_branchpoints_from_skel(skel,endnodes,branchpoints):
    for b in branchpoints:
        x,y=b
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                skel[x+i,y+j]=False
    return skel


#######Parallel version of find_segments

def find_skeleton_segments(skel: np.ndarray, endnodes:List[Tuple[int]], min_length:int,padding:int) -> (List[Part],np.ndarray):
    """Find stem parts between nodes using the connectivity in the skeleton.
    Returns a list of parts (pairs of nodes) with a minimum distance and a cleaned skeleton.        
    """
    t = Timer()
    t.start()
    out_skel=np.full(skel.shape,False)
    skeleton_parts=[]  

    def return_callback(result):
        nonlocal skeleton_parts
        nonlocal out_skel
        if result is not None:  
            skel_part, sub_skel, low_bounds, up_bounds  = result               
            skeleton_parts.append(skel_part) 
            out_skel[low_bounds[0]:up_bounds[0]+1,low_bounds[1]:up_bounds[1]+1]=sub_skel|out_skel[low_bounds[0]:up_bounds[0]+1,
                                                                                                  low_bounds[1]:up_bounds[1]+1]
                   
    def error_callback(error):
        print(error, flush=True)
                          
    print("#######################################################") 
    print("Find connected segments in the skeleton")                             
    print("Initial length of skeleton: ", np.count_nonzero(skel))      
    print("Number of endnodes", len(endnodes))
    print("Minimum length in pixel: ", min_length)
      
    pool = mp.Pool(mp.cpu_count()-1)
    r=[]    
    for endnode in endnodes:
        low_bounds=(endnode[0]-padding, endnode[1]-padding)
        up_bounds=(endnode[0]+padding, endnode[1]+padding)      
        sub_skel=skel[low_bounds[0]:up_bounds[0]+1,low_bounds[1]:up_bounds[1]+1]
        r.append(pool.apply_async(get_segment, args=(endnode,endnodes, sub_skel, low_bounds, up_bounds,min_length),
                                  callback=return_callback, error_callback=error_callback)) 
    for r_ in r:
         r_.wait()   
    pool.close()                   
    skeleton_parts=set(skeleton_parts) 
    print("Detected skeleton segments: ",len(skeleton_parts))   
    t.stop()
    print("#######################################################")   
    print("")  
                
    return skeleton_parts, out_skel
    

def get_segment(endnode,endnodes,skel, low_bounds, up_bounds, min_length):
    endnode=(endnode[0]-low_bounds[0],endnode[1]-low_bounds[1])
    for i in range(len(endnodes)):
        endnodes[i]=(endnodes[i][0]-low_bounds[0],endnodes[i][1]-low_bounds[1])
        
    temp_skel=np.full(skel.shape,False)
    x, y=endnode
    endnodes.remove(endnode) 
    skel[(x, y)]=False
    temp_skel[(x, y)]=True
    node=False
    l_bound_x=endnode[0]
    l_bound_y=endnode[1]
    u_bound_x=endnode[0]
    u_bound_y=endnode[1]
    length=0
    
    if len(get_neighbors(x,y,skel))==0:
        print("exit",flush=True)
        return None
    while node==False:
        frontier = get_neighbors(x, y, skel)  
        if frontier:     
            length=length+1
            if len(frontier)>1:
                print("dow where are all the neighbours from")
                print(endnode)
                print(x,y)
                print(frontier)
            x, y = frontier[0]
            if x<l_bound_x:
                l_bound_x=x
            if y<l_bound_y:
                l_bound_y=y
            if x>u_bound_x:
                u_bound_x=x
            if y>u_bound_y:
                u_bound_y=y
            skel[(x,y)]=False
            temp_skel[(x,y)]=True
            if frontier[0] in endnodes:
                node=True
                l_bound=(l_bound_x,l_bound_y)
                u_bound=(u_bound_x,u_bound_y)
                new_part = Part(endnode, frontier[0], [endnode, frontier[0]],l_bound,u_bound)               
        else:
            l_bound=(l_bound_x,l_bound_y)
            u_bound=(u_bound_x,u_bound_y)
            new_part = Part(endnode, (x,y), [endnode, (x,y)],l_bound,u_bound)
            node=True
    if length<min_length:
        return None
    
    if new_part.start[0]>new_part.stop[0]:
            new_part = Part(new_part.stop,new_part.start,[new_part.stop,new_part.start],l_bound,u_bound)    
            
    new_part.start=(new_part.start[0]+low_bounds[0],new_part.start[1]+low_bounds[1])
    new_part.stop=(new_part.stop[0]+low_bounds[0],new_part.stop[1]+low_bounds[1])
    for i in range(len(new_part.path)):
        new_part.path[i]=(new_part.path[i][0]+low_bounds[0],new_part.path[i][1]+low_bounds[1])
    new_part.l_bound=(new_part.l_bound[0]+low_bounds[0],new_part.l_bound[1]+low_bounds[1])
    new_part.u_bound=(new_part.u_bound[0]+low_bounds[0],new_part.u_bound[1]+low_bounds[1])
            
    return new_part, temp_skel,low_bounds, up_bounds


#######Parallel version of refine_skeleton_segments

def refine_skeleton_segments(parts:List[Part], skel: np.ndarray, distance:int)  -> (List[Part],np.ndarray):
    """Find stem parts between nodes using the connectivity in the skeleton.
    Returns a list of parts (pairs of nodes) with a minimum distance and a cleaned skeleton.        
    """
    split=0
    out=0
    refined_parts=[]  

    def return_callback(result):
        refined_part,s,o=result
        nonlocal split
        nonlocal out
        nonlocal refined_parts
        split=split+s
        out=out+o
        if refined_part is not None:
            for r in refined_part:         
                refined_parts.append(r)
                                        
    def error_callback(error):
        print(error, flush=True)
                          
    t = Timer()
    t.start()
    refined_parts=[]  
    
    print("#######################################################")    
    print("#Refining and sorting out skeleton segments")                                  
    print("Initial length of skeleton: ", np.count_nonzero(skel))      
    print("Number of initial skeleton segments", len(parts))
    
    pool = mp.Pool(mp.cpu_count()-1)
    r=[]
    for part in parts:
        low_bounds=(part.l_bound[0]-5,part.l_bound[1]-5)
        up_bounds=(part.u_bound[0]+5,part.u_bound[1]+5)
        sub_skel=skel[low_bounds[0]:up_bounds[0]+1,low_bounds[1]:up_bounds[1]+1]
                   
        r.append(pool.apply_async(refine_skeleton_segment, args=(part, low_bounds,up_bounds, sub_skel, distance),
                                  callback=return_callback, error_callback=error_callback)) 
    
    for r_ in r:
         r_.wait()     
    pool.close()
       
    print("Number of split segments:",split)   
    print("Number of removed segments:",out)
    print("Number of refined segments:",len(refined_parts))           

    t.stop()
    print("#######################################################") 
    print("")    
    return refined_parts    

def refine_skeleton_segment(part:Part,low_bounds:Tuple[int, int], up_bounds:Tuple[int, int],
                            skel: np.ndarray, distance:int) -> List[Part]:

        part.start=(part.start[0]-low_bounds[0],part.start[1]-low_bounds[1])
        part.stop=(part.stop[0]-low_bounds[0],part.stop[1]-low_bounds[1])
        part.path=[part.start,part.stop]
        refined_parts_=[]
        parts=[]
        parts.append(part)
        out_=0
        split_=0
        while len(parts)>0:

            w=parts[0].start
            n=parts[0].start
            z=parts[0].stop

            #p_last=part.path
            p_last=[parts[0].start,parts[0].stop]
            parts[0].path=[]
            parts[0].path.extend([w])
            temp=np.full(skel.shape, False)
            x_last,x_last=w
            while w != z:
                x, y = w
                w_=w
                skel[(x,y)]=False
                temp[(x,y)]=True
                ww = get_neighbors(x, y, skel)
                if ww:    
                    if len(ww)>1:
                        print("ww>1, should never happen", flush=True)
                        print("w_: ",w_, flush=True)
                        for w_2 in ww:                         
                            print(w_2,flush=True)
                            print("!!!",flush=True)
                            
                    #Step foreward        
                    w=ww[0]
                    p_recent=[n,w]
                    angle=ang(p_recent,p_last)

                   
                    
                    if w==z:
                        if angle >10:
                            new_part=Part(n,parts[0].stop,[n,parts[0].stop],low_bounds,up_bounds)
                            parts.append(new_part)
                            parts[0].stop=n
                            #restore points since last node
                            skel[np.where(temp==True)]=True
                            temp=np.full(skel.shape, False)
                            split_=split_+1
                        else:
                            parts[0].path.extend([w]) 
                           # x_,y_=w
                            temp=np.full(skel.shape, False)                          
                    else:
                        if math.dist(n, w)>distance:
                            if n==parts[0].start:
                                if angle >10:
                                    new_part=Part(w,parts[0].stop,[w,parts[0].stop],low_bounds,up_bounds)
                                    parts.append(new_part)
                                    parts[0].stop=w
                                    parts[0].path.extend([w])
                                    split_=split_+1
                                    z=w
                                else:                                    #Add node for diameter measurement
                                    parts[0].path.extend([w])
                                    p_last=p_recent
                                    n=w
                                    temp=np.full(skel.shape, False)
                                                                       
                            else:            
                                if angle >30:
                                    new_part=Part(n,parts[0].stop,[n,parts[0].stop],low_bounds,up_bounds)
                                    parts.append(new_part)
                                    parts[0].stop=n
                                    #restore points since last node
                                    skel[np.where(temp==True)]=True
                                    z=w    
                                    split_=split_+1  
                                    
                                else:   
                                    #Add node for diameter measurement
                                    parts[0].path.extend([w])
                                    p_last=p_recent
                                    n=w
                                    temp=np.full(skel.shape, False)

                else:
                    print("no neighbours found", flush=True)  
                    print(parts[0])
                    parts[0].path.extend([(x,y)])
                    parts[0].stop=(x,y)
                    print(parts)                
                    z=(x,y)
                    w=z
                
                

            refined_part_=Part(parts[0].start,parts[0].stop, parts[0].path,low_bounds,up_bounds)
            parts.pop(0)

            if math.dist(refined_part_.start,refined_part_.stop)>=distance:
                refined_part_.start=(refined_part_.start[0]+low_bounds[0],refined_part_.start[1]+low_bounds[1])
                refined_part_.stop=(refined_part_.stop[0]+low_bounds[0],refined_part_.stop[1]+low_bounds[1])
                for i in range(len(refined_part_.path)):
                    refined_part_.path[i]=(refined_part_.path[i][0]+low_bounds[0],refined_part_.path[i][1]+low_bounds[1])
                    
                if refined_part_.start[0]>refined_part_.stop[0]:           
                    refined_part_ = Part(refined_part_.stop,refined_part_.start,refined_part_.path,low_bounds,up_bounds)
                    refined_part_.path.reverse()
                    
                refined_parts_.append(refined_part_)
            else:
                out_=out_+1   
                                         
        if len(refined_parts_) ==0:
            return None, split_, out_
 
        return refined_parts_, split_, out_ 

#######Helper functions for skeleton operations   
    
def get_neighbors(x, y, skel) -> List[Tuple[int, int]]:
    #Returns the neighbours of a point(x,y) in the skeleton as list of coordinate tuples    
    width, height = skel.shape
    nb=[]
    for dy in (-1, 0, 1):
        cy = y + dy
        if cy < 0 or cy >= height:
            continue
        for dx in (-1, 0, 1):
            cx = x + dx
            if (dx != 0 or dy != 0) and 0 <= cx < width and skel[cx, cy]:
                nb.append((cx, cy))
    return nb

def create_vector(line):
    #Creates a normalized vecor from LineStrings or Tulple[Tuple[int]] 
    if type(line)=='LineString':
        v= [line.coords[-1][0]-line.coords[0][0], line.coords[-1][1]-line.coords[0][1]]
    else:
        v= [(line[1][0]-line[0][0]), (line[1][1]-line[0][1])]   
    return v/(np.linalg.norm(v)+epsilon)

def create_vector_org(line):
    #Creates a vecor from LineStrings or Tulple[Tuple[int]] 
    if type(line)=='LineString':
        return [line.coords[-1][0]-line.coords[0][0], line.coords[-1][1]-line.coords[0][1]]
    else:
        return [(line[1][0]-line[0][0]), (line[1][1]-line[0][1])]   
    return v
                     
def ang(lineA, lineB):
    #Calculates the angle between 2 vectors
    vA = create_vector(lineA)
    vB = create_vector(lineB)
    dot_product = np.dot(vA, vB)
    dot_product=np.clip(dot_product, -1, 1)
    angle = np.arccos(dot_product)
    ang_deg = np.degrees(angle)%380
    if ang_deg > 180:
        ang_deg = ang_deg-360
    return ang_deg
