import time
import sys
import pickle
import numpy as np
import os
import random
import cv2
import pandas as pd
import datetime
import traceback # Importieren Sie traceback hier, da es im __main__ Block verwendet wird
from functions2 import * # Angenommen, dies beinhaltet PrePipeline, TrackingDetectingMergeProcess, ROIPooling, SizeBasedDepthPredection, discard_outliers_and_find_expectation
from dataset import Dataset
from metrics import ROIDetectionEvaluation,DetectionMeasurements, AverageRelativeError, RMSE, MAE, empirical_cdf, MAEAtEachSection
from tqdm import tqdm
from tsmoothie.smoother import KalmanSmoother
import matplotlib.pyplot as plt
# import from metrics * ist hier redundant, da die Funktionen bereits einzeln importiert wurden
    

def test(testdata_path, depth_model = None, range_model = None,range_model2=None):
    """the testing function for the detector and range/depth estimator

    Args:
        testdata_path (_type_): A list of the test file names where the files are in Dataset folder
        depth_model (_type_, optional): the saved trained depth model path (all the models are in the Models/ folder). Defaults to None.
        range_model (_type_, optional): the saved trained range model path (all the models are in the Models/ folder). Defaults to None.
        range_model2 (_type_, optional): the saved trained range2 (using the same input of the depth model while output the estimated range) model path (all the models are in the Models/ folder). Defaults to None.

    Returns:
        dictionary: the results of the detector and the estimators.
    """
    # detector configuration
    expansion_coefficient = 20
    temperature_upper_bound = 37
    valid_region_area_limit = 5
    ROIevaluationThreshold = 0.5
    prepipeline = PrePipeline(expansion_coefficient, temperature_upper_bound)
    detector = TrackingDetectingMergeProcess(expansion_coefficient, valid_region_area_limit)

    # estimator configuration
    Roi_Pooling_Size = (2,4)
    resize_shape = (100 * Roi_Pooling_Size[0], 100 * Roi_Pooling_Size[1]) 
    window_size = 100
    roipooling = ROIPooling(resize_shape, window_size,window_size)
    topk = Roi_Pooling_Size[0] * Roi_Pooling_Size[1] # Korrigierte Berechnung (falls topk > Roi_Pooling_Size*2 ist)
    kalman_smoother = KalmanSmoother(component='level_trend', 
                                    component_noise={'level':0.1, 'trend':0.0000001})

    ## dataset loading
    testset = Dataset(testdata_path)

    # Model Loading (mit Existenzprüfung für robustere Initialisierung)
    range_estimator = pickle.load(open(range_model, 'rb')) if range_model and os.path.exists(range_model) else None
    range_estimator2 = pickle.load(open(range_model2, 'rb')) if range_model2 and os.path.exists(range_model2) else None
    depth_estimator = pickle.load(open(depth_model, 'rb')) if depth_model and os.path.exists(depth_model) else None
    
    TruePositive = []
    FalsePositive = []
    FalseNegtive = []


    Outputs = {
        'ira_matrix': [],
        'frame_index': [],
        'GT_range': [],  # Sequenziell (pro erkanntes Objekt)
        'GT_depth': [],  # Sequenziell (pro erkanntes Objekt)
        
        # --- KRITISCH: Daten pro erkanntem Objekt (Sequenziell) ---
        'GT_timestamps': [],        
        'Predicted_BBoxes': [],     # Sequenziell (flache Liste aller BBoxes, wenn sie gematcht wurden)
        'Object_IDs': [],           
        
        # --- KRITISCH: NEUER SCHLÜSSEL FÜR DIE VISUALISIERUNG (Pro Frame) ---
        'valid_BBoxes_per_frame': [],   # Liste von Listen: [[Boxen in Frame 1], [Boxen in Frame 2], ...]
        
        'depth_raw_prediction': [], 
        'depth_KF_smoothed_prediction': [], 
        'depth_Size_based_predictioins': [], 
        'depth_KF_smoothed_Size_based_predictioins': [], 
        'range_raw_prediction': [], 
        'range_KF_smoothed_prediction': [],
        'range2_raw_prediction': [], 
        'range2_KF_smoothed_prediction': [],
    }

    buffer_size = 10
    # Initialisierung der Buffer (bereinigt, um keine '0' als Schlüssel zu verwenden, wenn ID-Management stattfindet)
    buffer_pred_range = {}
    buffer_pred_range2 = {}
    buffer_pred_depth = {}
    buffer_pred_WH = {}
    buffer_pred_depth_final = {}

    # for sample_index in tqdm(range(testset.len())):
    for sample_index in range(testset.len()):
        ira_matrix, ambient_temperature, timestamps, GT_bbox, GT_depth, GT_range = testset.GetSample(sample_index)
        ira_img, subpage_type, ira_mat = prepipeline.Forward(ira_matrix, ambient_temperature)
        
        valid_BBoxes_in_frame = []

        if not isinstance(ira_img, (np.ndarray)):
            # 🛑 Wenn prepipeline fehlschlägt, speichern wir leere Daten für Konsistenz
            TP, FP, FN = ROIDetectionEvaluation(GT_bbox, [], [], threshold=ROIevaluationEvaluation) # Dummy Evaluation
            TruePositive.append(TP)
            FalsePositive.append(FP)
            FalseNegtive.append(FN)

        else:
            mask, x_split_mask_colored, filtered_mask_colored, prvs_mask_colored, original_BBoxes,original_timers, valid_BBoxes, valid_timers =  detector.Forward(ira_img)
            
            # 🛑 KRITISCH: Sammeln der erkannten BBoxes für diesen Frame
            valid_BBoxes_in_frame = valid_BBoxes 
            
            result, matched_bbox = ROIDetectionEvaluation(GT_bbox, valid_BBoxes,valid_timers, threshold=ROIevaluationThreshold)
            
            for i_box,ele in enumerate(matched_bbox):
                box, pred_box,count, id, index, IoU = ele
                x,y,w,h = pred_box
                if w == 0:
                    continue

                # 🛑 Sequenzielle Speicherung (pro erkanntes Objekt)
                Outputs['GT_timestamps'].append(timestamps)
                Outputs['Predicted_BBoxes'].append(pred_box) # Dies ist die flache Liste der BBoxes
                Outputs['Object_IDs'].append(id)
                Outputs['frame_index'].append(sample_index)
                Outputs['GT_range'].append(GT_range[index])
                Outputs['GT_depth'].append(GT_depth[index]) # Nur hier, da es an die Vorhersagen gebunden ist
                
                # --- Features extrahieren ---
                range_ = GT_range[index]
                depth = GT_depth[index]
                temp_roi = ira_mat[int(y):int(y+h), int(x):int(x+w)]
                try:
                    pooled_roi = roipooling.PoolingNumpy(temp_roi)
                except:
                    continue
                flat_data = np.reshape(np.array([pooled_roi]), (1, -1))
                sort_flat_data = np.sort(flat_data, axis=-1)[:,::-1]
                center_pont = (x+w/2,y+h/2)
                center_data = np.reshape(np.array([center_pont]), (1, -1))
                depth_estimater_input = np.concatenate((sort_flat_data[:,:topk],center_data),axis=1)
                
                
                # --- Range estimation (Range 1) ---
                range_final_output = None
                if range_estimator is not None:
                    predict_r = range_estimator.predict(sort_flat_data[:,:topk])[0]
                    Outputs['range_raw_prediction'].append(predict_r)
                    
                    str_id = str(id)
                    if str_id not in buffer_pred_range or count == 0:
                        buffer_pred_range[str_id] = [predict_r]
                        Outputs['range_KF_smoothed_prediction'].append(predict_r)
                        range_final_output = predict_r
                    else:
                        temp_predict = buffer_pred_range[str_id] + [predict_r]
                        buffer_pred_range[str_id].append(predict_r)  
                        kalman_smoother.smooth(temp_predict)
                        kf_predict = kalman_smoother.smooth_data[0][-1]
                        Outputs['range_KF_smoothed_prediction'].append(kf_predict)
                        range_final_output = kf_predict
                else:
                    Outputs['range_raw_prediction'].append(None)
                    Outputs['range_KF_smoothed_prediction'].append(None) # Platzhalter
                
                
                # --- Range2 estimation ---
                range2_final_output = None
                if range_estimator2 is not None:
                    predict_r = range_estimator2.predict(depth_estimater_input)[0]
                    Outputs['range2_raw_prediction'].append(predict_r)
                    
                    str_id = str(id)
                    if str_id not in buffer_pred_range2 or count == 0:
                        buffer_pred_range2[str_id] = [predict_r]
                        Outputs['range2_KF_smoothed_prediction'].append(predict_r)
                        range2_final_output = predict_r
                    else:
                        temp_predict = buffer_pred_range2[str_id] + [predict_r]
                        buffer_pred_range2[str_id].append(predict_r)  
                        kalman_smoother.smooth(temp_predict)
                        kf_predict = kalman_smoother.smooth_data[0][-1]
                        Outputs['range2_KF_smoothed_prediction'].append(kf_predict)
                        range2_final_output = kf_predict
                else:
                    Outputs['range2_raw_prediction'].append(None)
                    Outputs['range2_KF_smoothed_prediction'].append(None) # Platzhalter
                
                # --- Depth estimation ---
                depth_final_output = None
                if depth_model is not None:
                    predict_d = depth_estimator.predict(depth_estimater_input)[0]
                    Outputs['depth_raw_prediction'].append(predict_d)
                    
                    str_id = str(id)
                    if str_id not in buffer_pred_depth or count == 0:
                        # Initialisierung der Buffer für neue Objekte
                        buffer_pred_depth[str_id] = [predict_d]
                        buffer_pred_WH[str_id] = [[w,h]]
                        buffer_pred_depth_final[str_id] = [predict_d]

                        Outputs['depth_KF_smoothed_prediction'].append(predict_d)
                        Outputs['depth_Size_based_predictioins'].append([predict_d for i in range(buffer_size)])
                        Outputs['depth_KF_smoothed_Size_based_predictioins'].append(predict_d)                        
                    else:
                        buffer_pred_depth[str_id].append(predict_d)
                        buffer_pred_WH[str_id].append([w,h])
                        
                        # KF-Glättung
                        kalman_smoother.smooth(buffer_pred_depth[str_id])
                        kf_predict = kalman_smoother.smooth_data[0][-1]
                        Outputs['depth_KF_smoothed_prediction'].append(kf_predict)
                        
                        # Size-basierte Schätzung
                        size_based_predictions = SizeBasedDepthPredection(
                            kalman_smoother.smooth_data[0], # KF Smoothed History
                            buffer_pred_WH[str_id], 
                            buffer_size
                        )
                        Outputs['depth_Size_based_predictioins'].append(size_based_predictions)
                        
                        # Finaler Output (KF + Size-based)
                        temp_predict_list = [kf_predict] + size_based_predictions
                        depth_final_output, _ = discard_outliers_and_find_expectation(np.array(temp_predict_list))
                        Outputs['depth_KF_smoothed_Size_based_predictioins'].append(depth_final_output)
                else:
                    Outputs['depth_raw_prediction'].append(None)
                    Outputs['depth_KF_smoothed_prediction'].append(None)
                    Outputs['depth_Size_based_predictioins'].append([None for i in range(buffer_size)])
                    Outputs['depth_KF_smoothed_Size_based_predictioins'].append(None)
            
            # --- Detection Evaluation (nur für Frames mit ira_img) ---
            TP, FP, FN = result
            TruePositive.append(TP)
            FalsePositive.append(FP)
            FalseNegtive.append(FN)

        # 🛑 KRITISCH: SPEICHERN DER METADATEN PRO FRAME
        # Dies wird IMMER einmal pro Frame ausgeführt, auch wenn keine Boxen erkannt wurden.
        Outputs['ira_matrix'].append(ira_matrix)
        Outputs['valid_BBoxes_per_frame'].append(valid_BBoxes_in_frame) # Liste der BBoxes für diesen Frame

    Outputs['TruePositive'] = TruePositive
    Outputs['FalsePositive'] = FalsePositive
    Outputs['FalseNegtive'] = FalseNegtive

    return Outputs

        
if __name__ == "__main__":
    # ... (Rest des Codes bleibt unverändert) ...
    
    test_file_pathes = [
    'Dataset/Bathroom1_0_sensor_1.pickle',
    # ... (Alle anderen Pfade)
    ]

    # ... (Initialisierung von Results_save_path, Models und Schleife) ...
    
    Results_save_path = 'Outputs/'
    if not os.path.exists(Results_save_path):
        os.makedirs(Results_save_path)

    depth_model = 'Models/hgbr_depth.sav'
    range_model = 'Models/hgbr_range.sav'
    range_model2 = 'Models/hgbr_range2.sav'
    
    if not os.path.exists(depth_model) or not os.path.exists(range_model) or not os.path.exists(range_model2):
        print('WARNUNG: Eines oder mehrere Modelle (.sav) wurden nicht gefunden. Der Test kann fehlschlagen oder keine Schätzungen durchführen.')

    failed_file = []
    for file_name in test_file_pathes:
        try:
            print(f"\nStarting process: {file_name}")
            sys.stdout.flush()
            testdata_path = [file_name,]
            
            output = test(testdata_path, depth_model = depth_model, range_model = range_model, range_model2 =range_model2)
            
            output_filename = file_name.split('/')[-1].split('.')[0]
            with open(Results_save_path + output_filename + '.pkl', 'wb') as f:
                pickle.dump(output, f)
                print(f'✅ OUTPUT SAVED: {output_filename}.pkl') 
            
            TP = np.sum(np.array(output['TruePositive']))
            FP = np.sum(np.array(output['FalsePositive']))
            FN = np.sum(np.array(output['FalseNegtive']))
            precision, recall, F1_score = DetectionMeasurements(TP, FP, FN)
            print(f"ROI detection Performance: Precision={precision:.4f}, Recall={recall:.4f}, F1={F1_score:.4f}")
            
            sys.stdout.flush()

        except KeyboardInterrupt:
            print("\nKeyboard interrupt. Stopping processing.")
            break
        except Exception as e:
            print(f"\n🛑 FATAL ERROR processing {file_name}: {e}")
            # Korrigiere: traceback muss importiert werden
            # traceback.print_exc(file=sys.stdout)
            failed_file.append(file_name)
            pass
            
    print('\nfailed_file: ', failed_file)