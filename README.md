# Progetto di Ottimizzazione, Intelligenza Artificiale e Machine Learning
## Assegnazione dei compiti
**La seguente è la suddivisione indicativa dei compiti in quanto ogni parte del progetto sarà comunque seguita da tutti i membri del gruppo**
### 1. Analisi del Dataset & Annotazioni
**Responsabile: Dominici**

- Esaminare le immagini e verificare che le classi siano bilanciate (quante immagini per sasso, carta, forbici)

- Generare statistiche su distribuzione, dimensioni immagini, numero classi


### 2. Preprocessing & DataLoader personalizzato
**Responsabile: Dominici**

- Preprocessing delle immagini (resize, normalizzazione,...)

- Creare un DataLoader custom con PyTorch 

- Eventuale Data augmentation 


### 3. Modellazione rete
**Responsabile: Spitaleri**

- Progettare la rete neurale

- Costruire pipeline di training 


### 4. Training, Validazione, Salvataggi
**Responsabile: Seck**

- Gestire ciclo di training con logging delle metriche

- Monitoraggio con TensorBoard e scelta del modello migliore

### 5. Testing del modello addestrato
**Responsabile: Seck**
  
- Valutazione su immagini mai viste 

- Grafici metriche: Accuracy, Confusion Matrix, mAP 

- (opzionale) Visualizzare bounding box e classe predetta su immagini


# Relazione
https://colab.research.google.com/drive/1nGFvBGkIZX9lJMoHn_MFDnnVWvUJeCpP
## introduzione
L’obiettivo del progetto è addestrare un modello di visione artificiale in grado di analizzare un’immagine e riconoscere un feed continuo (Esempio: Webcam). Il modello, data in input un’immagine, deve essere in grado di classificare correttamente la mano tra tre possibili gesti: “sasso”, “carta” o “forbici”. Si tratta quindi di un problema di **image classification** con tre classi ben distinte, in cui l’oggetto da riconoscere è una mano in diverse posizioni.
L'obbiettivo infine è di ottenere un modello sufficientemente ottimizzato da poter essere eseguito all'interno di un’applicazione mobile, per permettere all’utente di giocare a “Sasso, carta, forbici” in tempo reale contro l'applicazione. 
## Descrizione dataset
## Architettura e scelte tecniche
## Risultati
## Riflessione sui problemi e soluzioni
### Membri del gruppo
- Jacopo Maria Spitaleri
- Alessandro Dominici
- Seck Mactar Ibrahima
