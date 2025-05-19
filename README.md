# Progetto di Ottimizzazione, Intelligenza Artificiale e Machine Learning
## Assegnazione dei compiti
### 1. Analisi del Dataset & Annotazioni
**Responsabile: Dominici**

- Esaminare le immagini e il file annotations.csv

- Verificare che le classi siano bilanciate (quante immagini per sasso, carta, forbici)

- Generare statistiche su distribuzione, dimensioni immagini, numero classi

- Preparare grafici esplicativi (Matplotlib/Seaborn)\
Output: exploration.py + grafici


### 2. Preprocessing & DataLoader personalizzato
**Responsabile: Seck**

- Implementare il preprocessing delle immagini (resize, normalizzazione)

- Creare un DataLoader custom con PyTorch o TensorFlow

- Dividere dataset in training/validation/test

- (Opzionale) Aggiungere tecniche di data augmentation (flip, zoom…)\
Output: dataloader.py, preprocessing.py

### 3. Modellazione (Transfer Learning)
**Responsabile: Spitaleri**

- Scegliere un modello preaddestrato (es. YOLOv5 o MobileNet SSD)

- Modificare l’output finale per 3 classi ("rock", "paper", "scissors")

- Costruire pipeline di training con configurazione in JSON\
Output: model.py, config.json, config_schema.json

### 4. Training, Validazione, Salvataggi
**Responsabile: Seck**

- Gestire ciclo di training con logging delle metriche
Implementare:

- Early stopping

- Salvataggio modello migliore

- Ripresa da checkpoint

- Monitoraggio con TensorBoard (facoltativo ma consigliato)\
Output: train.py, train.log, runs/ (per TensorBoard)

### 5. Testing del modello addestrato
**Responsabile: Spitaleri**

- Caricare il modello salvato

- Valutarlo su immagini mai viste (test set)

- Stampare metriche: Accuracy, Confusion Matrix, mAP (se possibile)

- Visualizzare bounding box e classe predetta su immagini\
Output: test.py, metrics_report.txt, output_images/

### 6. GUI o demo di inferenza
**Responsabile: Dominici**

- Creare una piccola interfaccia (anche via terminale o Streamlit)

- Permettere all’utente di caricare un’immagine e ottenere predizione

- Mostrare l’immagine con bounding box e nome del gesto ("Sasso", ecc.)\
Output: demo.py o app.py

# Relazione
## introduzione
L’obiettivo del progetto è addestrare un modello di visione artificiale in grado di analizzare un’immagine e riconoscere la posizione e il tipo di oggetti presenti, con particolare riferimento alla mano umana. In questo caso specifico, l’applicazione è finalizzata alla realizzazione di un sistema capace di identificare il gesto compiuto dalla mano all’interno del gioco “Sasso, carta, forbici”. Il modello, dato in input un’immagine, deve essere in grado di classificare correttamente la mano tra tre possibili gesti: “sasso”, “carta” o “forbici”. Si tratta quindi di un problema di **image classification** con tre classi ben distinte, in cui l’oggetto da riconoscere è una mano umana in diverse posizioni.
Infine, il modello potrà essere integrato in un’applicazione interattiva, ad esempio tramite webcam, per permettere all’utente di giocare a “Sasso, carta, forbici” in tempo reale contro la macchina.
## Descrizione dataset
## Architettura e scelte tecniche
## Risultati
## Riflessione sui problemi e soluzioni
### Membri del gruppo
- Jacopo Maria Spitaleri
- Alessandro Dominici
- Seck Mactar Ibrahima
