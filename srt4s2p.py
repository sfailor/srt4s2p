# Functions for aligning suite2p ROIs between experiments

from os.path import join,isfile
import glob
import numpy as np
import cv2 as cv
import key_point_finder as kp
from scipy.stats import mode
import matplotlib.pyplot as plt
import pandas as pd
import pyqtgraph as pg
from PyQt5 import QtCore


def load_results(file, expt_num, true_only=True):
    # Loads match file and returns aligned spk and stat 
    
    matches = np.load(file,allow_pickle=True)[()]
    rec_dirs = matches['rec_dirs']
    matches = matches['matches']

    if true_only:
        matches = matches[matches.true_match]

    roi_cols = [c for c in matches.columns if 'roi' in c]
    plane_cols = [c for c in matches.columns if 'plane' in c]
    matches.sort_values(by=plane_cols, inplace = True)
       
    stat = []
    spks = []
       
    for i in range(len(plane_cols)):
        stat.append(np.zeros(len(matches),dtype='object'))
        r0 = 0
        for n,p in enumerate(matches[plane_cols[i]].unique()):
            ind = matches[plane_cols[i]] == p
            rois = matches.loc[ind,roi_cols[i]]
            plane_stat = np.load(join(rec_dirs[i],f'plane{p}','stat.npy'), allow_pickle=True)[rois]
            plane_spks = np.load(join(rec_dirs[i],f'plane{p}','spks.npy'), mmap_mode='r')[rois]
            if n == 0:
                spks.append(np.zeros((len(matches),plane_spks.shape[1])))
            
            stat[i][ind] = plane_stat
            spks[i][ind,:] = plane_spks
            
            r0 += len(plane_stat)
                    
    return spks, stat


class Compare2p():
    
    def __init__(self):    
        self.recordings = {}
        self.comparisons = {}
        self.matches = {}
        
    def add_recording(self,rec_dir,ids=None):
        
        if not isinstance(rec_dir,(tuple,list,np.ndarray)):
            rec_dir = [rec_dir]
        
        if not all(isinstance(i,str) for i in rec_dir):
            raise TypeError('Recording directory or directories must be strings')
        
        if ids == None:
            print('ID(s) were not provided. The recording directory, or directories, will be the ID(s).')
            ids = rec_dir
        
        if not isinstance(ids,(tuple,list,np.ndarray)):
            ids = [ids]
        if not all(isinstance(i,(str,int)) for i in ids):
            raise TypeError('ID(s) must be a string or int.')
                       
        if len(rec_dir) == len(ids):
            for d,i in zip(rec_dir,ids):
                self.recordings[i] = {'rec_dir' : d}
        else:
            raise ValueError('There must be a corresponding ID for each recording.')
        
        return self
    
    def remove_recording(self,ids):
        
        if not isinstance(ids,(tuple,list,np.ndarray)):
            ids = [ids]
        for i in ids:
            if i in self.recordings.keys():
                del self.recordings[i]
            else:
                raise ValueError('Recording ID is not found!')
        
        return self
            
    def __recording_info(self):
        info = ''
        for k in self.recordings.keys():
                info += f"""
                        recording ID: {k}
                        suite2p directory: {self.recordings[k]['rec_dir']}
                        """
        return info
    
    def list_recordings(self):
        print(self.__recording_info())
        
    def __str__(self):
      return self.__recording_info()
    
    def compare_planes(self,ids,planes,layers = None):
        
        if not isinstance(ids,(tuple,list,np.ndarray)):
            ids = [ids]
        
        # Check ids exist
        id_present = [i in self.recordings.keys() for i in ids]
        
        if False in id_present:
            raise ValueError(f'ID {ids[id_present.index(False)]} not found!')
                
        if layers == None:
            layers = ['meanImg','meanImgE',['meanImgE','Vcorr'],['meanImgE','roi']]
        
        print('Loading ops files.')
        for i,p in zip(ids,planes):
            if p not in self.recordings[i].keys():
                self.recordings[i][p] = {'ops' : self.load_suite2p(self.recordings[i]['rec_dir'],p,'ops.npy')}
            elif 'ops' not in self.recordings[i][p].keys():
                self.recordings[i][p]['ops'] = self.load_suite2p(self.recordings[i]['rec_dir'],p,'ops.npy')
            
        ops = [self.recordings[i][p]['ops'] for i,p in zip(ids,planes)]
        
        if any('roi' in l for l in layers):
            
            print('Loading stat files.') 
            for i,p in zip(ids,planes):
                if 'stat' not in self.recordings[i][p].keys():  
                    self.recordings[i][p]['stat'] = self.load_suite2p(self.recordings[i]['rec_dir'],p,'stat.npy')
            print('Loading iscell files.')
            for i,p in zip(ids,planes):
                if 'iscell' not in self.recordings[i][p].keys():   
                    self.recordings[i][p]['iscell'] = self.load_suite2p(self.recordings[i]['rec_dir'],p,'iscell.npy')
            
            stat = [self.recordings[i][p]['stat'] for i,p in zip(ids,planes)]
            iscell = [self.recordings[i][p]['iscell'] for i,p in zip(ids,planes)]
            
            built_layers = [self.build_layers(o, layers, s, i) for o,s,i in zip(ops, stat, iscell)]
        else:
            built_layers = [self.build_layers(o, layers) for o in ops]
            
        labels = [f'{i} plane {p}' for i,p in zip(ids,planes)]
        
        layer_labels = []
                
        for l in layers:
            if type(l) is str:
                layer_labels.append(l)
            else:
                label = ''
                for i in range(len(l)):
                    if i == 0:
                        label = label + f'{l[i]}'
                    else:
                        label = label + f' and {l[i]}'
                    
                layer_labels.append(label)
                
        results = kp.compare_images(built_layers, labels, [layer_labels]*len(ids))
        
        key = (tuple(i for i in ids),tuple(p for p in planes))
        
        # Save only if key points were made and there are the same number for each plane
        if all(results['points']) and len(results['points'][0]) == len(results['points'][1]):
            self.comparisons[key] = results
        else:
            print("Either no points were choosen or there are one or more unpaired points.\nNo results saved.")
        
        return self
    
    def find_manual_matches(self,ids, planes='all'):
        ids = tuple(i for i in ids)
        ids_key = [(ids[0], i) for i in ids[1:]]
        
        df_matches = pd.DataFrame()
            
        for i_key in ids_key:
        
            if planes == 'all':
                print(f'Finding manually matched ROIs for all plane comparisons for id pair {i_key}.')
                plane_key = [k[1] for k in self.comparisons.keys() if k[0] == i_key]
            elif isinstance(planes,(tuple,list,np.ndarray)) and len(planes) == 2:
                print(f'Finding manually matched ROIs for planes {planes} of id pair {i_key}')
                plane_key = [tuple(p for p in planes)]
            else:
                raise TypeError("Argument 'planes' should either be 'all' or a pair of plane numbers in an interable)")
            
            df_pair = pd.DataFrame()
            
            for pk in plane_key:
            
                comp_key = (i_key,pk)
                
                if comp_key not in self.comparisons.keys():
                    raise ValueError('Comparison between recordings has not been made!')
                
                for i,p in zip(i_key,pk):
                    if 'roi_map' not in self.recordings[i][p].keys():
                        ops = self.recordings[i][p]['ops'] 
                        stat = self.recordings[i][p]['stat']
                        iscell = self.recordings[i][p]['iscell']
                        self.recordings[i][p]['roi_map'],_ = self.make_roi_map(stat,iscell,ops['meanImg'].shape)
        
                points0,points1 = self.comparisons[(i_key,pk)]['points']
                points0,points1 = np.array(points0),np.array(points1)
                matches = np.array(self.comparisons[(i_key,pk)]['point_high_conf']).T
                
                # Check for any manual matches
                if ~np.any(matches):
                    print('No manual matches')
                    return self
                
                points0 = np.round(points0[np.all(matches,axis=1)]).astype(int)
                points1 = np.round(points1[np.all(matches,axis=1)]).astype(int)
                
                rois = []
                
                for i in range(len(points0)):
                    roi0 = self.recordings[i_key[0]][pk[0]]['roi_map'][points0[i,0],points0[i,1]]
                    roi1 = self.recordings[i_key[1]][pk[1]]['roi_map'][points1[i,0],points1[i,1]]
                    
                    if roi0 != 0 and roi1 != 0:
                       rois.append([roi0-1, roi1-1])
                       
                rois = np.array(rois)                   
                
                df_pair = pd.concat([df_pair, pd.DataFrame({f'rois_{i_key[0]}' : rois[:,0].astype(int),
                                                            f'rois_{i_key[1]}' : rois[:,1].astype(int),
                                                            f'plane_{i_key[0]}' : np.repeat(pk[0],len(rois)).astype(int),
                                                            f'plane_{i_key[1]}' : np.repeat(pk[1],len(rois)).astype(int),
                                                            'overlap' : np.ones(len(rois)),
                                                            'true_match' : np.ones(len(rois)).astype(bool)})], ignore_index=True)
                
            # Remove duplicate matches
            df_pair = self.clean_matches(df_pair)
            
            if len(df_matches) == 0:
                df_matches = pd.concat([df_matches,df_pair])
                df_matches.set_index([f'rois_{ids[0]}',f'plane_{ids[0]}'], inplace=True)
            else:
                df_pair.set_index([f'rois_{ids[0]}',f'plane_{ids[0]}'], inplace=True)
                df_matches = pd.concat([df_matches,df_pair], sort=False, axis=1)
            
        if ids in self.matches.keys():
            df_matches.reset_index(inplace=True)
            self.matches[ids] = pd.concat([self.matches[ids],df_matches],axis=0, ignore_index=True)
            self.matches[ids].drop_duplicates(inplace=True)
        else:
            self.matches[ids] = df_matches.reset_index()
            
        self.matches[ids].sort_values(by=['overlap',f'plane_{ids[0]}'], ascending=[False,True], ignore_index=True, inplace=True)
        
        return self

    def list_comparisons(self):
        text = ''
        for k in list(self.comparisons.keys()):
            recording = k[0]
            planes = k[1]
            if 'overlapping_rois' in self.comparisons[k]:
                overlapping_rois = True
            else:
                overlapping_rois = False
            text += f"""
                     Recordings: {recording}
                     Planes: {planes}
                     """
        print(text)
        
    def find_transform(self, ids, planes='all', method='perspective'):

        ids = tuple(i for i in ids)
        ids_key = [(ids[0], i) for i in ids[1:]]


        if planes == 'all':
                print(f'Finding transform matrix M for all comparisons of id pair {ids}.')
                plane_key = [k[1] for k in self.comparisons.keys() if k[0] == ids]
        elif isinstance(planes,(tuple,list,np.ndarray)) and len(planes) == 2:
                print(f'Finding transform matrix M for comparison of {planes} of id pair {ids}')
                plane_key = [tuple(p for p in planes)]      
        
        for i_key in ids_key:         

            for pk in plane_key:
                key = (i_key,pk)
                
                ops = [self.recordings[i][p]['ops'] for i,p in zip(i_key,pk)]
                stat = [self.recordings[i][p]['stat'] for i,p in zip(i_key,pk)]
                iscell = [self.recordings[i][p]['iscell'] for i,p in zip(i_key,pk)]
                    
                if key not in self.comparisons.keys():
                    raise ValueError('Comparison between recordings has not been made!')
            
                self.comparisons[key]['M'] = self.find_M(self.comparisons[key]['points'][0],self.comparisons[key]['points'][1],method)
                self.comparisons[key]['trans_method'] = method
            
        return self
    
    def transform_map(self, ids, planes, image = 'meanImg'):
        key = (tuple(i for i in ids),tuple(p for p in planes))
        if (tuple(ids),tuple(planes)) in self.comparisons.keys():
            if 'M' not in self.comparisons[key]:
                raise ValueError(f'Calculate transform matrix M before attempting transformations. To do this run find_transform({ids})')
            
            if image.lower() == 'roi' or image.lower() == 'rois':
                ref,_ = self.make_roi_map(self.recordings[ids[0]][planes[0]]['stat'],
                                        self.recordings[ids[0]][planes[0]]['iscell'],
                                        self.recordings[ids[0]][planes[0]]['ops']['meanImg'].shape)
                ref = ref > 0 
                other,_ = self.make_roi_map(self.recordings[ids[1]][planes[1]]['stat'],
                                        self.recordings[ids[1]][planes[1]]['iscell'],
                                        self.recordings[ids[1]][planes[1]]['ops']['meanImg'].shape)
                other = other > 0
                other_trans,_ = self.make_transformed_roi_map(self.recordings[ids[1]][planes[1]]['stat'],
                                                              self.recordings[ids[1]][planes[1]]['iscell'],
                                                              self.recordings[ids[0]][planes[0]]['ops']['meanImg'].shape,
                                                              self.comparisons[key]['M'],
                                                              self.comparisons[key]['trans_method'])
                other_trans = other_trans > 0
            elif image.lower() == 'vcorr':
                ref = self.pad_Vcorr(self.recordings[ids[0]][planes[0]]['ops'])
                other = self.pad_Vcorr(self.recordings[ids[1]][planes[1]]['ops'])
                other_trans = self.transform_image(other, ref.shape, self.comparisons[key]['M'], self.comparisons[key]['trans_method'])
            else:
                try:
                    ref = self.recordings[ids[0]][planes[0]]['ops'][image]
                    other = self.recordings[ids[1]][planes[1]]['ops'][image]
                    other_trans = self.transform_image(other, ref.shape, self.comparisons[key]['M'], self.comparisons[key]['trans_method'])
                except ValueError:
                    print("image argument must either be 'rois' or an image from ops.npy (e.g., 'meanImg')")
        else:
            raise ValueError('Key points have not been found for this comparison!')
        
        f,a = plt.subplots(1,3)
        a[0].imshow(ref)
        a[0].set_title(ids[0])
        a[1].imshow(other_trans)
        a[1].set_title(f'{ids[1]} transformed')
        a[2].imshow(other)
        a[2].set_title(f'{ids[1]} original')
    
        for a in a.flatten():
            a.set_xticks([])
            a.set_yticks([])
    
    def find_overlapping_rois(self,ids,planes='all',threshold=0.6):
        
        ids = tuple(i for i in ids)
        ids_key = [(ids[0], i) for i in ids[1:]]
        
        df_matches = pd.DataFrame()
            
        for i_key in ids_key:
        
            if planes == 'all':
                print(f'Finding overlapping ROIs for all plane comparisons for id pair {i_key}.')
                plane_key = [k[1] for k in self.comparisons.keys() if k[0] == i_key]
            elif isinstance(planes,(tuple,list,np.ndarray)) and len(planes) == 2:
                print(f'Finding overlapping ROIs for planes {planes} of id pair {i_key}')
                plane_key = [tuple(p for p in planes)]
            else:
                raise TypeError("Argument 'planes' should either be 'all' or a pair of plane numbers in an interable)")
            
            df_pair = pd.DataFrame()
            
            for pk in plane_key:
            
                comp_key = (i_key,pk)
                
                if comp_key not in self.comparisons.keys():
                    raise ValueError('Comparison between recordings has not been made!')
                elif 'M' not in self.comparisons[comp_key]:
                    raise ValueError(f'Calculate transform matrix M before attempting transformations. To do this run find_transform({ids})')
                
                ops = [self.recordings[i][p]['ops'] for i,p in zip(i_key,pk)]
                stat = [self.recordings[i][p]['stat'] for i,p in zip(i_key,pk)]
                iscell = [self.recordings[i][p]['iscell'] for i,p in zip(i_key,pk)]
                
                # Make rois map if it don't already exist
                
                if 'roi_map' not in self.recordings[i_key[0]][pk[0]].keys():
                    self.recordings[i_key[0]][pk[0]]['roi_map'],_ = self.make_roi_map(stat[0],iscell[0],ops[0]['meanImg'].shape)
                
                # This is currently generated every time, in case a new transformation was calculated
                self.comparisons[comp_key]['roi_map_trans'],_ = self.make_transformed_roi_map(stat[1], iscell[1], ops[0]['meanImg'].shape, 
                                                                                              self.comparisons[comp_key]['M'], 
                                                                                              self.comparisons[comp_key]['trans_method'])
                
                rois, overlap = self.find_overlapping(self.recordings[i_key[0]][pk[0]]['roi_map'], self.comparisons[comp_key]['roi_map_trans'], threshold)
                plane_nums = np.repeat(np.array(pk)[None,:], len(rois), axis = 0)
                true_match = np.ones(len(rois)).astype(bool) # For keep track during curation
                
                df_pair = pd.concat([df_pair, pd.DataFrame({f'rois_{i_key[0]}' : rois[:,0].astype(int),
                                                            f'rois_{i_key[1]}' : rois[:,1].astype(int),
                                                            f'plane_{i_key[0]}' : plane_nums[:,0].astype(int),
                                                            f'plane_{i_key[1]}' : plane_nums[:,1].astype(int),
                                                            'overlap' : overlap,
                                                            'true_match' : true_match})], ignore_index=True)
            
            # Remove duplicate matches
            df_pair = self.clean_matches(df_pair)
        
            if len(df_matches) == 0:
                df_matches = pd.concat([df_matches,df_pair])
                df_matches.set_index([f'rois_{ids[0]}',f'plane_{ids[0]}'], inplace=True)
            else:
                df_pair.set_index([f'rois_{ids[0]}',f'plane_{ids[0]}'], inplace=True)
                df_matches = pd.concat([df_matches,df_pair], sort=False, axis=1)
                
        if ids in self.matches.keys():
            df_matches.reset_index(inplace=True)
            self.matches[ids] = pd.concat([self.matches[ids],df_matches],axis=0, ignore_index=True)
            self.matches[ids].drop_duplicates(inplace=True)
        else:
            self.matches[ids] = df_matches.reset_index()
             
        self.matches[ids].sort_values(by=['overlap',f'plane_{ids[0]}'], ascending=[False,True], ignore_index=True, inplace=True)
        
        return self
                                    
    def inspect_matches(self,ids):
        
        ids_key = tuple(i for i in ids)
      
        if ids_key not in self.matches.keys():
            raise ValueError(f'Comparisons for id pair not found!')
        
        planes = self.matches[ids_key][[f'plane_{ids[0]}',f'plane_{ids[1]}']].to_numpy()
        
        # pregenerate maps
        
        mean_maps = {i : {} for i in ids}        
        mean_maps_e = {i : {} for i in ids}
        roi_maps = {i : {} for i in ids}
        roi_maps_binary = {i : {} for i in ids}
        vcorr_maps = {i : {} for i in ids}
        
        print('Generating maps... (mean image, Vcorr, etc.)')
               
        for p in np.unique(planes[:,0]):
            
            map_shape = self.recordings[ids[0]][p]['ops']['meanImg'].shape
            
            mean_maps[ids[0]][p] = self.recordings[ids[0]][p]['ops']['meanImg']
            mean_maps_e[ids[0]][p] = self.recordings[ids[0]][p]['ops']['meanImgE']

            if 'roi_map' not in self.recordings[ids[0]][p].keys():
                self.recordings[ids[0]][p]['roi_map'] = self.make_roi_map(self.recordings[ids[0]][p]['stat'], self.recordings[ids[0]][p]['iscell'],map_shape)[0]
                
            roi_maps[ids[0]][p] = self.recordings[ids[0]][p]['roi_map']
            roi_maps_binary[ids[0]][p] = self.recordings[ids[0]][p]['roi_map'] > 0
            
            vcorr_maps[ids[0]][p] = self.pad_Vcorr(self.recordings[ids[0]][p]['ops'])
            
        for p in np.unique(planes,axis=0):
            
            p = tuple(p)
            
            map_shape = self.recordings[ids[0]][p[0]]['ops']['meanImg'].shape
            
            mean_maps[ids[1]][p] = self.transform_image(self.recordings[ids[1]][p[1]]['ops']['meanImg'], 
                                                        map_shape,
                                                        self.comparisons[(ids_key,tuple(p))]['M'],
                                                        self.comparisons[(ids_key,tuple(p))]['trans_method'])
            
            mean_maps_e[ids[1]][p] = self.transform_image(self.recordings[ids[1]][p[1]]['ops']['meanImgE'], 
                                                          map_shape,
                                                          self.comparisons[(ids_key,tuple(p))]['M'],
                                                          self.comparisons[(ids_key,tuple(p))]['trans_method'])
               
            if 'roi_map_trans' not in self.comparisons[(ids_key,tuple(p))].keys():

                self.comparisons[(ids_key,p)]['roi_map_trans']= self.make_transformed_roi_map(self.recordings[ids[1]][p[1]]['stat'],
                                                                                              self.recordings[ids[1]][p[1]]['iscell'], 
                                                                                              map_shape,
                                                                                              self.comparisons[(ids_key,tuple(p))]['M'],
                                                                                              self.comparisons[(ids_key,tuple(p))]['trans_method'])[0]
                
            roi_maps[ids[1]][p] = self.comparisons[(ids_key,tuple(p))]['roi_map_trans']
            roi_maps_binary[ids[1]][p] = self.comparisons[(ids_key,tuple(p))]['roi_map_trans'] > 0
            
            if 'Vcorr_trans' not in self.comparisons[(ids_key,tuple(p))].keys():
                self.comparisons[(ids_key,p)]['Vcorr_trans'] = self.transform_image(self.pad_Vcorr(self.recordings[ids[1]][p[1]]['ops']),
                                                                                     map_shape,
                                                                                     self.comparisons[(ids_key,tuple(p))]['M'],
                                                                                     self.comparisons[(ids_key,tuple(p))]['trans_method'])
            
            vcorr_maps[ids[1]][p] = self.comparisons[(ids_key,p)]['Vcorr_trans']
                   
            
        layers = [mean_maps,mean_maps_e,vcorr_maps,roi_maps_binary]
       
       
        app = pg.mkQApp()
        app.setQuitOnLastWindowClosed(True)    
        
        screen_resolution = app.desktop().screenGeometry()
        width, height = screen_resolution.width(), screen_resolution.height()

        width = int(width*0.75)
        height = int(width/2)
        
        win = ROIView(self.matches[ids], roi_maps, layers, (width,height))

        app.exec()
    
        self.matches[ids_key]['true_match'] = win.true_matches
        
        return self
    
        
    def save_matches(self, ids, save_path = None):
        
        ids = tuple(i for i in ids)
        if save_path == None:
            save_dir = self.recordings[ids[0]]['rec_dir']
            
            save_file = 'matches_for'
            for i in ids:
                save_file = save_file + f'_{i}'
                
            save_path = join(save_dir,save_file)
        
        
        results = {'matches' : None, 'rec_dirs' : None}
        results['matches'] = self.matches[ids]
        results['rec_dirs'] = [self.recordings[i]['rec_dir'] for i in ids]
        
                
        np.save(save_path, results)
        
       
    @staticmethod            
    def plane_dir(rec_dir,plane):
        return join(rec_dir,f'plane{plane}')
    
    @staticmethod
    def load_suite2p(rec_dir,plane,file_type='ops.npy'):
        return np.load(join(Compare2p.plane_dir(rec_dir,plane),file_type),allow_pickle=True)[()]
                             
    @staticmethod
    def make_roi_map(stat,iscell,shape=(512,512)):
        
        roi_map = np.zeros(shape=shape)
        
        rois_mapped = []
            
        for i in range(len(stat)):
        
            if iscell[i,0] == 1:
            
                ypix = stat[i]['ypix'][~stat[i]['overlap']]
                xpix = stat[i]['xpix'][~stat[i]['overlap']]
                
                if len(ypix) != 0:
                    roi_map[ypix,xpix] = i+1
                    rois_mapped.append(i+1)
                
        return roi_map, rois_mapped

    @staticmethod
    def pad_Vcorr(ops):
        
        shape = ops['meanImg'].shape
        
        return np.pad(ops['Vcorr'], ((ops['yrange'][0],shape[0]-ops['yrange'][1]), (ops['xrange'][0],shape[1]-ops['xrange'][1])))

    @staticmethod
    def build_layers(ops, layers = ['meanImg','meanImgE', ['meanImgE','Vcorr'],['meanImgE','roi']], stat = None, iscell = None):

        shape = ops['meanImg'].shape

        layer_dict = {}

        if stat is None or iscell is None:
            if 'roi' in layers: 
                layers.remove('roi')
                print('To include an ROI layer you must provide a stat dict and iscell array')
        else:
            if True in ['roi' in l for l in layers]:
                roi_map,_ = Compare2p.make_roi_map(stat,iscell,shape)
                layer_dict['roi'] = (roi_map > 0) * 0.5 # Change this to adjust how bright ROIs are relative to mean image, etc.

        uni_layers = list(np.unique(np.concatenate([np.unique(l) for l in layers])))
        uni_layers.remove('roi')

        for l in uni_layers:
            if l == 'Vcorr':
                layer_dict[l] = Compare2p.pad_Vcorr(ops)
            else:
                layer_dict[l] = ops[l]

        built_layers = []
        
        for l in layers:
            
            if type(l) is str:
                l = [l]
                layer_image = np.zeros(shape+(1,))
            elif len(l) >= 2:
                layer_image = np.zeros(shape+(3,))
            
            if len(l) > 3:
                print('Only three maps to a layer (final is RGB). The first three will be used.')
                l = [l[i] for i in range(3)]
            
            for i in range(len(l)):
                layer_image[...,i] = layer_dict[l[i]]

            layer_image = np.squeeze(layer_image) # Remove extra dim if image isn't RGB

            layer_image = cv.normalize(layer_image, None, 0, 1, cv.NORM_MINMAX) # Normalize between 0 and 1

            built_layers.append(layer_image)           
                       
        return built_layers

    @staticmethod
    def cv_points(points):
        return np.array([[p[1],p[0]] for p in points])

    @staticmethod
    def find_M(pts1, pts2, method='perspective'):
        
        pts1 = Compare2p.cv_points(pts1)
        pts2 = Compare2p.cv_points(pts2)

        if method == 'perspective':
            M,_ = cv.findHomography(pts2,pts1,cv.RANSAC)
        elif method == 'similarity':
            M,_ = cv.estimateAffinePartial2D(pts2,pts1)
            
        return M

    @staticmethod
    def transform_image(image,shape,M,method='perspective'):
        
        if method == 'perspective':
            return cv.warpPerspective(image, M, dsize = shape, flags = cv.INTER_LINEAR)
        elif method == 'similarity':
            return cv.warpAffine(image, M, dsize = shape, flags = cv.INTER_LINEAR)

    @staticmethod
    def transform_points(pts,M,method='perspective'):
        # Points should already be in CV preferred format
            
        if method == 'perspective':
            return np.squeeze(cv.perspectiveTransform(pts,M))
        elif method == 'similarity':
            return np.squeeze(cv.transform(pts,M))

    @staticmethod
    def make_transformed_roi_map(stat, iscell, shape, M, method = 'perspective'):
        
        roi_map = np.zeros(shape=shape)
        
        rois_mapped = []
            
        for i in range(len(stat)):
        
            if iscell[i,0] == 1:
            
                ypix = stat[i]['ypix'][~stat[i]['overlap']]
                xpix = stat[i]['xpix'][~stat[i]['overlap']]
                
                if len(ypix) != 0:
                    
                    # Make point array the way openCV wants it
                    pts = np.array([[x,y] for x,y in zip(xpix,ypix)], dtype = 'float32')
                    pts = np.array([pts])
                    
                    new_pts = Compare2p.transform_points(pts,M,method)

                    if len(new_pts.shape) > 1: # Don't include ROIs with only a single pixel
                    
                        ypix,xpix = np.round(new_pts[:,1]).astype(int),np.round(new_pts[:,0]).astype(int)

                        good_pix = np.logical_and(ypix >= 0,ypix <= shape[0]-1) & np.logical_and(xpix>= 0,xpix <= shape[0]-1)
                        
                        if len(good_pix) > 3: # At least 3 pixel needs to fall in the imaged area of the reference

                            ypix = ypix[good_pix]
                            xpix = xpix[good_pix]

                            # dialate and erode to fill in any gaps
                            dilated = np.zeros(shape)
                            dilated[ypix,xpix] = 1
                            dilated = cv.dilate(dilated.astype(float),np.ones((3,3),dtype='uint8'),iterations=1)
                            eroded = cv.erode(dilated.astype(float),np.ones((3,3),dtype='uint8'))
                            
                            ypix,xpix = np.nonzero(eroded)
                                    
                            roi_map[ypix,xpix] = i+1
                            rois_mapped.append(i+1)
                
        return roi_map, rois_mapped

    @staticmethod
    def find_overlapping(roi_map_1, roi_map_2, threshold=0.6): # Finds overlapping ROIs based on some threshold
        
        overlapping_rois = []
        overlap_prop = []

        uni_r1 = np.unique(roi_map_1)
        uni_r1 = uni_r1[uni_r1!=0] # Remove background value

        for r1 in uni_r1:
            ypix,xpix = np.nonzero(roi_map_1==r1)
            r2_vals = roi_map_2[ypix,xpix].ravel()
            r2_vals = r2_vals[r2_vals!=0] # Remove background
            if len(r2_vals) > 0:
                best_r2 = mode(r2_vals, keepdims=False)[0] # Find roi from expt2 that is most overlapping
                mask1 = roi_map_1==r1
                mask2 = roi_map_2==best_r2
                overlap = np.sum(mask1*mask2)/np.max([np.sum(mask1),np.sum(mask2)]) # Overlap as proportion of the largest ROI
                if overlap > threshold:
                    overlapping_rois.append([r1-1,best_r2-1]) # Change back to zero-based indexing so it aligns with stat.npy
                    overlap_prop.append(overlap)

        overlapping_rois = np.array(overlapping_rois)
        overlapping_rois = overlapping_rois[np.argsort(overlap_prop)[::-1],:]

        return overlapping_rois, np.sort(np.array(overlap_prop))[::-1]
    
    @staticmethod
    def clean_matches(df):
        # Remove any duplicates
        df.drop_duplicates(inplace=True)
        
        # Find any cases where an ROI has been matched more than once, and keep the match with the largest overlap
        n_recordings = int((df.shape[1]-1)/2) # Number of recordings in the df. There is an roi and plane column for each recording.
        
        for n in range(n_recordings):
            ref_cols = df.columns[[n,n+n_recordings]] # roi and plane column names
            mask = df.duplicated(subset=ref_cols, keep=False)
            duplicates = df[mask]
            max_ind = [duplicates.loc[np.all(duplicates[ref_cols]==r,axis=1),'overlap'].idxmax() for _,r in duplicates[ref_cols].drop_duplicates().iterrows()]
            df = pd.concat([df[~mask],duplicates.loc[max_ind]], ignore_index=True)
        
        return df
              
class ROIView(pg.GraphicsLayoutWidget):
        
    pg.setConfigOptions(imageAxisOrder='row-major')
    pg.setConfigOption('background', (0,0,0))
    
    def __init__(self, matches, roi_maps, views, size):
        pg.GraphicsLayoutWidget.__init__(self, size=size, title=None, show=True, border=False)
        roi_columns = [c for c in matches.columns if 'rois' in c]
        self.rois = matches[roi_columns].to_numpy()
        plane_columns = [c for c in matches.columns if 'plane' in c]
        self.planes = matches[plane_columns].to_numpy()
        self.overlap = np.round(matches.overlap.to_numpy(),3)
        self.roi_maps = roi_maps
        self.views = views
        self.true_matches = matches.true_match.to_numpy()
        self.counter = 0
        self.view = 0
        
        self.v0 = self.addViewBox(row=0, col=0, invertY=True, enableMenu=False, lockAspect=True, enableMouse=True)
        self.v1 = self.addViewBox(row=0, col=1, invertY=True, enableMenu=False, lockAspect=True, enableMouse=True)
        
        self.bg0 = pg.ImageItem()
        self.r0 = pg.ImageItem()
        self.bg1 = pg.ImageItem()
        self.r1 = pg.ImageItem()
        
        self.v0.addItem(self.bg0)
        self.v0.addItem(self.r0)
        
        self.v1.addItem(self.bg1)
        self.v1.addItem(self.r1)
        
        bg_cmap = pg.colormap.getFromMatplotlib('Greys_r')
        roi_cmap = pg.colormap.ColorMap(pos=[0,1],color = [pg.mkColor(0,0,0,0),pg.mkColor(255,0,0,255)])
        
        self.bg0.setColorMap(bg_cmap)
        self.bg1.setColorMap(bg_cmap)
        self.r0.setColorMap(roi_cmap)
        self.r1.setColorMap(roi_cmap)
        
        self.r0.setZValue(10)
        self.r0.setOpacity(0.2)
        self.r1.setZValue(10)
        self.r1.setOpacity(0.2)
        
        self.label0 = pg.TextItem(anchor=(0,0))
        self.label1 = pg.TextItem(anchor=(0,0))
        
        self.v0.addItem(self.label0)
        self.v1.addItem(self.label1)
        
        self.v0.sigStateChanged.connect(self.update_labels)
        self.v1.sigStateChanged.connect(self.update_labels)
        
        self.scene().sigMouseClicked.connect(self.mouseClick)
        
        self.update()
    
    
    def update(self):
        
        roi0,roi1 = self.rois[self.counter]
        plane0,plane1 = self.planes[self.counter]
        
        roi0 = roi0 + 1 # roi map labels start at 1
        roi1 = roi1 + 1
        
        ids = list(self.roi_maps.keys())
        
        r0_cen = np.mean(np.array(np.nonzero(self.roi_maps[ids[0]][plane0]==roi0)),axis=1).astype(int)
        r1_cen = np.mean(np.array(np.nonzero(self.roi_maps[ids[1]][(plane0,plane1)]==roi1)),axis=1).astype(int)
               
        bg0_image = self.views[self.view][ids[0]][plane0]

        self.bg0.setImage(bg0_image, autoLevels = True)
        self.r0.setImage(self.roi_maps[ids[0]][plane0]==roi0, autoLevels = True)
        
        bg1_image = self.views[self.view][ids[1]][(plane0,plane1)]
        
        self.bg1.setImage(bg1_image, autoLevels = True)
        self.r1.setImage(self.roi_maps[ids[1]][(plane0,plane1)]==roi1, autoLevels = True)
        
        self.v0.setRange(xRange = (r0_cen[1] - 50, r0_cen[1] + 50), yRange = (r0_cen[0] - 50, r0_cen[0] + 50))
        self.v1.setRange(xRange = (r1_cen[1] - 50, r1_cen[1] + 50), yRange = (r1_cen[0] - 50, r1_cen[0] + 50))
        
        self.update_labels()
        
    def update_view(self):
        
        roi0,roi1 = self.rois[self.counter]
        plane0,plane1 = self.planes[self.counter]
        
        roi0 = roi0 + 1 # roi map labels start at 1
        roi1 = roi1 + 1
        
        ids = list(self.roi_maps.keys())
                
        bg0_image = self.views[self.view][ids[0]][plane0]

        self.bg0.setImage(bg0_image, autoLevels = True)
        self.r0.setImage(self.roi_maps[ids[0]][plane0]==roi0, autoLevels = True)
        
        bg1_image = self.views[self.view][ids[1]][(plane0,plane1)]
        
        self.bg1.setImage(bg1_image, autoLevels = True)
        self.r1.setImage(self.roi_maps[ids[1]][(plane0,plane1)]==roi1, autoLevels = True)
        
        self.update_labels()
        
        
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Q:
            self.view = 0
            self.update_view()
        elif event.key() == QtCore.Qt.Key_W:
            self.view = 1
            self.update_view()
        elif event.key() == QtCore.Qt.Key_E:
            self.view = 2
            self.update_view()
        elif event.key() == QtCore.Qt.Key_R:
            self.view = 3
            self.update_view()
        elif event.key() == QtCore.Qt.Key_A:
            self.counter = (self.counter -1) % len(self.rois)
            self.update()
        elif event.key() == QtCore.Qt.Key_D:
            self.counter = (self.counter + 1) % len(self.rois)
            self.update()
        elif event.key() == QtCore.Qt.Key_S:
            if self.true_matches[self.counter]:
                self.true_matches[self.counter] = False
            else:
                self.true_matches[self.counter] = True
            self.update()
    
    
    def mouseClick(self, event):
        if event.button() == QtCore.Qt.RightButton:
            self.counter = (self.counter + 1) % len(self.rois)
            self.update()
        elif event.button() == QtCore.Qt.LeftButton:
            self.counter = (self.counter - 1) % len(self.rois)
            self.update()
    
    
    def update_labels(self):
        self.label0.setHtml(self.text()[0])
        self.label1.setHtml(self.text()[1])
        
        view_range0 = self.v0.viewRange()
        view_range1 = self.v1.viewRange()
            
        self.label0.setPos(min(view_range0[0]),min(view_range0[1]))
        self.label1.setPos(min(view_range1[0]),min(view_range1[1]))
         
         
    def text(self):
        hex = '#C70039'
        text0 = f'<font size = "+6" color = "{hex}">Pair {self.counter} of {len(self.rois)-1}<br>Match set to <i>{self.true_matches[self.counter]}</i></font>'
        text1 = f'<font size = "+6" color = "{hex}">Cells {self.rois[self.counter,0]} and {self.rois[self.counter,1]}<br>Planes {self.planes[self.counter,0]} and {self.planes[self.counter,1]}<br>Overlap {self.overlap[self.counter]}</font>'

        return text0,text1
            

    
                
        