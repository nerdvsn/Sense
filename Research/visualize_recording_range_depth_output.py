import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
from matplotlib.patches import Rectangle 

# === WICHTIG: TKAGG-BACKEND FÜR VNC/TERMINAL AUSFÜHRUNG ===
try:
    mpl.use('TkAgg') 
except ImportError:
    print("Warnung: TkAgg-Backend nicht verfügbar. Bitte 'sudo apt install python3-tk' ausführen.")

# === PFAD ZUR DATENDATEI ANPASSEN ===
file_path = "Bathroom1_0_sensor_1.pkl"  

# === DEKLARATION DER BEZUGSAUFLÖSUNG ===
ORIGINAL_RGB_WIDTH = 640
ORIGINAL_RGB_HEIGHT = 480


def load_data(file_path):
    """
    Lädt alle benötigten Daten aus der Pickle-Datei.
    Verwendet die Schlüssel aus der Ausgabe Ihrer test.py.
    """
    print(f"Lade Daten aus: {file_path}...")
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # KRITISCH: Verwenden von 'valid_BBoxes_per_frame'
        REQUIRED_KEYS = ['ira_matrix', 'frame_index', 'GT_range', 'valid_BBoxes_per_frame', 'range_raw_prediction']
        
        if all(key in data for key in REQUIRED_KEYS):
            ira_matrix = np.array(data['ira_matrix'])
            frame_indices = np.array(data['frame_index']) 
            
            predicted_bboxes = data['valid_BBoxes_per_frame'] # <-- KORRIGIERTER SCHLÜSSEL
            
            range_raw_predictions = data.get('range_raw_prediction', [])
            gt_range = data.get('GT_range', []) 
            
            print(f"Erfolgreich geladen. Gesamt-Frames: {ira_matrix.shape[0]}")
            return ira_matrix, frame_indices, predicted_bboxes, range_raw_predictions, gt_range
        else:
            missing_keys = [key for key in REQUIRED_KEYS if key not in data]
            print(f"FEHLER: Einer oder mehrere benötigte Schlüssel fehlen in der Datei: {missing_keys}")
            return None, None, None, None, None
            
    except FileNotFoundError:
        print(f"FEHLER: Datei nicht gefunden unter {file_path}.")
        return None, None, None, None, None
    except Exception as e:
        print(f"FEHLER beim Laden oder Verarbeiten der Pickle-Datei: {e}")
        return None, None, None, None, None


def visualize_frames(ira_matrix, frame_indices, predicted_bboxes, range_predictions, gt_range):
    """Erstellt eine Matplotlib-Animation der thermischen Aufnahmen mit prädizierten Bounding Boxes und Range-Text."""
    if ira_matrix is None or ira_matrix.size == 0:
        print("Keine Frames zum Visualisieren vorhanden.")
        return
    
    IR_HEIGHT = ira_matrix.shape[1] 
    IR_WIDTH = ira_matrix.shape[2]  

    scale_x = 1.0 
    scale_y = 1.0

    fig, ax = plt.subplots(figsize=(10, 8))
    
    global_min = 15.0  
    global_max = 45.0  
    cmap = 'inferno' 
    
    therm_image = ax.imshow(
        ira_matrix[0], 
        cmap=cmap,
        vmin=global_min,
        vmax=global_max,
        interpolation='nearest' 
    )

    cbar = fig.colorbar(therm_image)
    cbar.set_label('Temperature [°C]', fontsize=14)
    
    ax.set_xticks(np.arange(0, IR_WIDTH, 4))
    ax.set_yticks(np.arange(0, IR_HEIGHT, 4))
    ax.set_xlabel(f'Horizontal Pixel ({IR_WIDTH})')
    ax.set_ylabel(f'Vertikal Pixel ({IR_HEIGHT})')
    
    rectangles = []
    texts = []
    
    # 🛑 KORREKTUR DER DEKLARATION: Initialisierung des Zählers im Geltungsbereich von visualize_frames
    prediction_counter = 0 

    def update_frame(i):
        """Funktion, die in jedem Animationsschritt aufgerufen wird."""
        # 🛑 KORREKTUR: Jetzt kann 'nonlocal' verwendet werden, da prediction_counter
        # in der äußeren (nonlocal) Funktion 'visualize_frames' deklariert wurde.
        nonlocal prediction_counter
        
        # 1. Bilddaten aktualisieren
        therm_image.set_data(ira_matrix[i]) 
        
        # 2. Alte Elemente entfernen
        for rect in rectangles:
            rect.remove()
        rectangles.clear()
        
        for txt in texts:
            txt.remove()
        texts.clear()
        
        # 3. Bounding Boxes und Range-Vorhersagen hinzufügen
        current_bboxes = predicted_bboxes[i] 
        
        if not isinstance(current_bboxes, list):
             current_bboxes = [] 

        num_bboxes_in_frame = len(current_bboxes)
        valid_boxes_count = 0

        for k in range(num_bboxes_in_frame):
            
            bbox = current_bboxes[k]
            
            # SICHERHEITSPRÜFUNG
            if not isinstance(bbox, (list, tuple, np.ndarray)) or len(bbox) != 4:
                print(f"WARNUNG Frame {i}, Box {k}: Ungültige BBox-Struktur: {bbox}. Überspringe.")
                
                prediction_counter += 1
                continue
                
            x_min, y_min, w, h = bbox 
            valid_boxes_count += 1
            
            # Range-Wert aus der sequenziellen Liste abrufen
            predicted_range = None
            if prediction_counter < len(range_predictions):
                predicted_range = range_predictions[prediction_counter]
            
            prediction_counter += 1 # Zähler hochzählen für das nächste Objekt

            # Rechteck hinzufügen
            rect = Rectangle(
                (x_min * scale_x, y_min * scale_y), 
                w * scale_x, h * scale_y,           
                linewidth=2,                    
                edgecolor='cyan',               
                facecolor='none'
            )
            ax.add_patch(rect)
            rectangles.append(rect)
            
            # Text-Anzeige
            text_label = f"R: {predicted_range:.2f} m" if predicted_range is not None else "R: N/A"
            
            # Positioniere den Text leicht oberhalb der Box
            txt = ax.text(
                x_min * scale_x, 
                y_min * scale_y - 1,
                text_label, 
                color='yellow', 
                fontsize=9, 
                bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=1)
            )
            texts.append(txt)
        
        # 4. Titel aktualisieren (dynamisch)
        updated_title_text = (
            f"Frame {i+1} von {ira_matrix.shape[0]} | Gefunden: {valid_boxes_count}"
        )
        title = ax.set_title(updated_title_text)
        
        # Rückgabe aller zu aktualisierenden Elemente
        return [therm_image, title] + rectangles + texts

    # Erstelle die Animation
    ani = animation.FuncAnimation(
        fig, 
        update_frame, 
        frames=ira_matrix.shape[0], 
        interval=100, 
        blit=False, 
        repeat=True
    )

    # Zeigt das Fenster an und startet die Animation
    plt.show()


# --- Hauptausführung ---
if __name__ == "__main__":
    ira_matrix, frame_indices, predicted_bboxes, range_predictions, gt_range = load_data(file_path)
    
    if ira_matrix is not None and predicted_bboxes is not None and range_predictions is not None:
        visualize_frames(ira_matrix, frame_indices, predicted_bboxes, range_predictions, gt_range)
    else:
        print("Daten konnten nicht vollständig geladen werden. Beende das Skript.")