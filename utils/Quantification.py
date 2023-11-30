#!/usr/bin/env python

##################################################################################
'''Imports'''

import numpy as np
from typing import List
import math
from shapely.geometry import Point, LineString
import rasterio.features
import geopandas as gpd
import multiprocessing as mp

from stand_alone.WINMOL_Analyzer import Stem
from stand_alone.WINMOL_Analyzer import Timer
from IO import get_bounds_from_profile

#System epsilon
epsilon = np.finfo(float).eps


                

################################################################################## 
'''Stem quantification operations'''

#####Parallel quantification of stem parameters

def quantify_stems(stems:List[Stem],pred, profile):
    #Quantification of the stem parameters
    t = Timer()
    t.start()
       
    stems_=[]
    stems__=[]

    print("#######################################################")   
    print("Quantifying stems")   
    stems=get_diameters(stems, pred, profile)   
    d_l=[s.d for s in stems]
    
    pool = mp.Pool(mp.cpu_count()-1)
    for stem in pool.imap_unordered(clean_diameter,stems):
        stems_.append(stem)

    for stem in pool.imap_unordered(quantify_stem,stems_):
        stems__.append(stem)  
    pool.close()

    print("Volume of ", len(stems__), " stems calculated")  
    t.stop()
    print("#######################################################")   
    print("")  
    return stems__

#####Parallel version of get_diameters

def get_diameters(stems:List[Stem], pred, profile):
    #Calculates the diameters for all stems in the list
    
    px_size=profile['transform'][0]
    bounds=get_bounds_from_profile(profile)
    transform=profile['transform']
    contours=[]
    
    mask=None
    pred[np.where( pred < 0.5 )] = 0
    pred[np.where( pred >= 0.5 )] = 1
    pred=pred.astype(np.int16)
    pred_shapes_ = ({'properties': {'raster_val': v}, 'geometry': s} for i, (s, v) in enumerate(rasterio.features.shapes(pred, mask=mask, transform=transform))) 
    pred_shapes = list(pred_shapes_)
    pred_shapes  = gpd.GeoDataFrame.from_features(pred_shapes)
    pred_shapes = pred_shapes[pred_shapes['raster_val']==1]

    diam_count=0
    measured_stems=[]    
    def return_callback(measured_stem):
        measured_stems.append(measured_stem)
        nonlocal diam_count
        diam_count=diam_count+len(measured_stem.d)
    def error_callback(error):
        print(error, flush=True)
    
    pool = mp.Pool(mp.cpu_count()-1)
    r=[]
    for stem in stems:
        r.append(pool.apply_async(calc_v_d, args=(stem, pred_shapes),callback=return_callback, error_callback=error_callback)) 
      
    for r_ in r:
        r_.wait()
    pool.close()
    
    print(diam_count," measurements of diameters where conducted")    
    return measured_stems    

def quantify_stem(stem):
    #Calculates the volume and length of a stem
    stem.l=[]
    stem.v=[]
    for i in range(0,len(stem.path.coords)-1):
        seg_l,seg_vol=calc_l_v(stem.path.coords[i],stem.path.coords[i+1], stem.d[i],stem.d[i+1])
        stem.l.append(seg_l)
        stem.v.append(seg_vol)
    stem.Volume=sum(stem.v)
    stem.Length=stem.start.distance(stem.stop)
    return stem   


#####Helper functions

def clean_diameter(stem):
    #Replaces outlier from the diameter list by interpolation or substitution
    Q1=np.quantile(stem.d, 0.25)
    Q3=np.quantile(stem.d, 0.75)
    IQR = Q3 - Q1
    LW = Q1 - 1.5*IQR
    UW = Q3 + 1.5*IQR  
    if len(stem.d)>4:
        for i in range(1,len(stem.d)-2):
            if stem.d[i]>UW or stem.d[i]<LW:
                wd1=stem.d[i-1]*abs(Point(stem.path.coords[i]).distance(Point(stem.path.coords[i+1])))
                wd2=stem.d[i+1]*abs(Point(stem.path.coords[i-1]).distance(Point(stem.path.coords[i])))
                d12=abs(Point(stem.path.coords[i-1]).distance(Point(stem.path.coords[i+1])))
                stem.d[i]=(wd1+wd2)/d12
        if stem.d[0]>UW or stem.d[0]<LW:
            stem.d[0]=stem.d[1]
        if stem.d[-1]>UW or stem.d[-1]<LW:
            stem.d[-1]=stem.d[-2]    
    return(stem)

def calc_v_d(stem, contours):       
    #calulate radial vector to mesure the diameter
    vector=create_vector((stem.path.coords[0],stem.path.coords[1]))
    vector=[-vector[1],vector[0]]
    p1=Point(stem.path.coords[0][0]-vector[0]*1.0,stem.path.coords[0][1]-vector[1]*1.0)
    p2=Point(stem.path.coords[0][0]+vector[0]*1.0,stem.path.coords[0][1]+vector[1]*1.0)
    vector=LineString([p1,p2]) 
    stem.d.append(calc_d(stem.path.coords[0],vector, contours))
    stem.vector.append(vector)
        
    for i in range(1,len(stem.path.coords)-1):
        vector=create_vector((stem.path.coords[i-1],stem.path.coords[i+1]))
        vector=[-vector[1],vector[0]]
        p1=Point(stem.path.coords[i][0]-vector[0]*1.0,stem.path.coords[i][1]-vector[1]*1.0)
        p2=Point(stem.path.coords[i][0]+vector[0]*1.0,stem.path.coords[i][1]+vector[1]*1.0)
        vector=LineString([p1,p2]) 
        stem.d.append(calc_d(stem.path.coords[i],vector, contours))
        stem.vector.append(vector)
            
    vector=create_vector((stem.path.coords[-2],stem.path.coords[-1]))
    vector=[-vector[1],vector[0]]
    p1=Point(stem.path.coords[-1][0]-vector[0]*1.0,stem.path.coords[-1][1]-vector[1]*1.0)
    p2=Point(stem.path.coords[-1][0]+vector[0]*1.0,stem.path.coords[-1][1]+vector[1]*1.0)
    vector=LineString([p1,p2]) 
    stem.d.append(calc_d(stem.path.coords[-1],vector, contours))
    stem.vector.append(vector)
      
    return stem
    
def calc_d(node,line,contours):
    #Calculate the diameter for a specific node
    node=Point(node)
    d=0
    intersects=contours.geometry.intersection(line)
    intersects=intersects[~intersects.is_empty]  

    for i in intersects:     
        if node.distance(i)< 0.01:
            if i.geom_type == 'MultiLineString':     
                for i_ in i.geoms:
                    if node.distance(i_)< 0.01:
                        d=i_.length
            else:
                d=i.length   
    return d


def calc_d__org(stem, contours):       
    #calulate radial vector to mesure the diameter
    vector=create_vector((stem.path.coords[0],stem.path.coords[1]))
    vector=[-vector[1],vector[0]]
    stem.d.append(calc_d(stem.path.coords[0],vector, contours))
    stem.vector.append(vector)
        
    for i in range(1,len(stem.path.coords)-1):
        vector=create_vector((stem.path.coords[i-1],stem.path.coords[i+1]))
        vector=[-vector[1],vector[0]]
        stem.d.append(calc_d(stem.path.coords[i],vector, contours))
        stem.vector.append(vector)
            
    vector=create_vector((stem.path.coords[-2],stem.path.coords[-1]))
    vector=[-vector[1],vector[0]]
    stem.d.append(calc_d(stem.path.coords[-1],vector, contours))
    stem.vector.append(vector)
    return stem
    
def calc_d_org(node,v,contours):
    #Calculate the diameter for a specific node
    p1=Point(node[0]-v[0]*1.0,node[1]-v[1]*1.0)
    p2=Point(node[0]+v[0]*1.0,node[1]+v[1]*1.0)
    node=Point(node)
    line=LineString([p1,p2])
    d=0
    intersects=contours.geometry.intersection(line)
    intersects=intersects[~intersects.is_empty]  

    for i in intersects:     
        if node.distance(i)< 0.01:
            if i.geom_type == 'MultiLineString':     
                for i_ in i.geoms:
                    if node.distance(i_)< 0.01:
                        d=i_.length
            else:
                d=i.length   
    return d

def calc_l_v(p1,p2,d1,d2):
    #Calculate the length and volume of a segment described by 2 points and the respective diameters
    l=math.dist(p1,p2)
    v=1/3*math.pi*((d1/2)**2+(d1/2)*(d2/2)+(d2/2)**2)*l
    return l,v    

#######Helper functions for skeleton operations   

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

