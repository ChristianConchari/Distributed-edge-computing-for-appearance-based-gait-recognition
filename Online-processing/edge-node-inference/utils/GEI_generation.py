import numpy as np
import cv2

def GEI_generator(sil_file, final_size=160, debug=False):
    stack_GEI = []
    lenfiles = len(sil_file)

    for idimg, img in enumerate(sil_file):
    
        size=128
        biggest = np.zeros_like(img)
        contours1, _ = cv2.findContours(img.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt=max(contours1, key=cv2.contourArea)
        cv2.drawContours(biggest, [cnt], -1, 255, -1)    
        
        if (len(contours1)>0):
            ncoun= np.concatenate(contours1)[:, 0, :]
            x1, y1 = np.min(ncoun, axis=0)
            x2, y2 = np.max(ncoun, axis=0)
            silhouette = biggest[y1:y2, x1:x2]
            
            
            factor = size/max(silhouette.shape)
            height = round(factor*silhouette.shape[0])
            width = round(factor*silhouette.shape[1])

            if(height>width):
                nor_sil = cv2.resize(silhouette,(width,height))
                portion_body = 0.3                                                    
                moments = cv2.moments(nor_sil[0:int(nor_sil.shape[0]*portion_body),])
                w = round(moments['m10']/(moments['m00']+1))
                background = np.zeros((final_size, final_size))
                shift = round((final_size/2)-w)
                if(shift<0 or shift+nor_sil.shape[1]>size):
                    shift = round((final_size-nor_sil.shape[1])/2)
                
                nor_sil = cv2.resize(nor_sil,(nor_sil.shape[1],final_size))
                cf = int(0.13*nor_sil.shape[1])
                if  not np.sum(nor_sil[:,nor_sil.shape[1]//4*2:nor_sil.shape[1]//4*3]) > np.sum(nor_sil[:,(nor_sil.shape[1]//3*2):]):
                    background[:,shift-cf:nor_sil.shape[1]+shift-cf] = nor_sil
                elif not np.sum(nor_sil[:,nor_sil.shape[1]//4*2:nor_sil.shape[1]//4*3]) > np.sum(nor_sil[:,:(nor_sil.shape[1]//3)]):
                    background[:,shift+cf:nor_sil.shape[1]+shift+cf] = nor_sil
                else:
                    background[:,shift:nor_sil.shape[1]+shift] = nor_sil
                    
                
                stack_GEI.append(background)

    if stack_GEI == []:
        GEI = np.zeros((final_size, final_size))
        print('\tNo Files Found')
    else:
        GEI = np.mean(np.array(stack_GEI), axis=0)

    return GEI, stack_GEI