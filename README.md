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


# Relazione Progetto: Ottimizzazione, Intelligenza Artificiale e Machine Learning

## Introduzione

L’obiettivo di questo progetto è la realizzazione di un sistema di visione artificiale, tramite tecniche di Deep Learning, per il riconoscimento dei gesti della mano corrispondenti alle classi “sasso”, “carta” e “forbici”. Il sistema è pensato per funzionare sia su immagini statiche sia in tempo reale da webcam, con particolare attenzione all’ottimizzazione del modello per velocità e robustezza.

---

## Descrizione del Dataset e Preprocessing

Il dataset utilizzato è strutturato in cartelle distinte per ciascuna classe (“rock”, “paper”, “scissors”) e per le suddivisioni training e test. L’analisi iniziale è stata fondamentale per verificare il bilanciamento tra le classi e la qualità delle immagini. A questo scopo sono stati generati grafici di distribuzione delle classi e preview di campioni casuali tramite il modulo `dataset_analysis.py`.

Le fasi di preprocessing sono state implementate per garantire uniformità e robustezza nel caricamento dei dati:
- Conversione delle immagini in scala di grigi e ridimensionamento a 50x50 pixel, scelta risultata la più efficace dopo test comparativi con segmentazioni HSV e blob analysis.
- Implementazione di un DataLoader custom (classe `RPSDataset` in `dataset.py`), in grado di scartare automaticamente immagini non processabili e mappare nomi delle classi in etichette numeriche.
- Data augmentation tramite la classe `Augmentations` (`dataset/augmentations.py`), applicando trasformazioni stocastiche come flip, rotazione, jitter di colore, affine e random erasing.

Esempio di preprocessing robusto:

```python
def preprocess(img, target_size=(50, 50)):
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, target_size)
    gray = np.expand_dims(gray, axis=0)
    return gray
```

L’intera pipeline garantisce la pulizia e l’omogeneità del dataset, prerequisito essenziale per il training efficace della rete.

---
## Monitoraggio e Analisi con TensorBoard e Moduli di Analisi
### Logging e Visualizzazione con TensorBoard

Il progetto integra un sistema di logging delle metriche di training e validazione tramite TensorBoard. Questo è reso possibile grazie a un modulo dedicato che, durante l’addestramento, registra in automatico:
- Loss e accuratezza per ogni epoca
- Valori delle metriche su validation set
- Eventuali immagini di esempio, confusion matrix e altri dati diagnostici

In questo modo, è possibile:
- Monitorare l’andamento del training e identificare rapidamente episodi di overfitting, underfitting o plateaux delle metriche
- Confrontare tra loro diversi esperimenti in modo visivo
- Analizzare graficamente i trend delle metriche per prendere decisioni data-driven su modifiche di modello, preprocessing o iperparametri

### Analisi Approfondita: la cartella `analysis`

Oltre a TensorBoard, la cartella `analysis` contiene script e notebook per l’analisi esplorativa del dataset e dei risultati:
- **Distribuzione delle classi:** script che producono grafici a barre della distribuzione delle classi nel dataset, utili per diagnosticare eventuali sbilanciamenti.
- **Anteprima immagini:** generazione di collage di campioni casuali per ogni classe, per una rapida ispezione visiva della qualità e varietà dei dati.
- **Metriche avanzate:** possibilità di calcolare e visualizzare confusion matrix, precision, recall e altre metriche aggregate, utilizzando i file di log prodotti durante il training/test.
- **Analisi degli errori:** strumenti per identificare esempi “difficili” e visualizzare le predizioni errate, facilitando il debugging del modello.

---

## Architettura e Scelte Tecniche

### Modello

Il modello principale è una CNN compatta, progettata ad hoc per input monocanale 50x50, implementata in `src/model.py`. La rete impiega tre blocchi convoluzionali con batch normalization e max pooling, seguiti da un classificatore fully connected. Sono state sperimentate anche varianti più leggere e diverse strategie di augmentation negli script della cartella `experiments/`.

```python
class CNN(nn.Module):
    def __init__(self, input_size=50, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

### Pipeline di Training

- Gestione training/validazione automatica, con salvataggio del modello migliore, scheduler del learning rate, early stopping e logging su TensorBoard.
- Ottimizzazione tramite Adam optimizer e CrossEntropyLoss (con varianti usando label smoothing).
- Augmentation avanzata per robustezza (flip, rotazioni, jitter, affine, random erasing e, in alcuni esperimenti, MixUp).
- Scheduler di learning rate con riduzione ogni 10 epoche.
- Configurazione centralizzata via file JSON (`config/config.json`), per riproducibilità e flessibilità degli esperimenti.

---

## Risultati

- Accuratezza di test costantemente superiore all’85-90%, variabile in base alla pipeline di preprocessing e augmentation adottata.
- Salvataggio automatico del modello con la migliore performance su validation set.
- Metriche di loss e accuracy loggate durante il training, con possibilità di visualizzazione tramite TensorBoard.
- Inferenza in tempo reale via webcam, con predizione visualizzata “live” e possibilità di test immediato della robustezza del modello.

---

## Riflessione sui Problemi e Soluzioni

- **Bilanciamento delle classi:** Analisi statistica iniziale e, nelle versioni avanzate, ponderazione della loss per classi sbilanciate.
- **Preprocessing robusto:** Testate diverse pipeline (HSV, blob, MediaPipe, scala di grigi); scelta finale della scala di grigi per semplicità, efficienza e generalizzabilità.
- **Overfitting:** Affrontato con augmentation aggressiva e regularizzazione (dropout, early stopping).
- **Early Stopping:** Introdotto per evitare eccessivo overfitting e rendere il training più efficiente.
- **Testing su webcam:** Pipeline dedicata per validare la performance del modello in scenari reali e visualizzare anche le maschere di segmentazione.

---

## Utilizzo avanzato: quantizzazione e export

### Quantizzazione

Il file `src/quantize.py` permette la quantizzazione del modello PyTorch addestrato, riducendone la dimensione e aumentando la velocità di inferenza, importante per deployment su hardware limitato.

### Esportazione

Il file `src/export.py` consente di esportare modelli in diversi formati (TorchScript, ONNX, custom), facilitando l’integrazione in applicazioni di produzione o dispositivi mobili.

---

## Esempio di utilizzo del modello con webcam

Il file `src/webcam.py` acquisisce video dalla webcam, preprocessa i frame in tempo reale, effettua l’inferenza e mostra la predizione sovrapposta. Questo consente una valutazione immediata del modello in condizioni realistiche.

---

## Caratteristiche del progetto

- **Codice modulare:** Suddiviso in moduli per dataset, augmentations, preprocessing, modello, training, inferenza, quantizzazione, export ed esperimenti.
- **Preprocessing e Data Augmentation:** Ampio uso di tecniche per aumentare la robustezza e la generalizzazione del modello.
- **Custom Dataset/DataLoader:** Gestione personalizzata compatibile con PyTorch.
- **Validazione, testing, early stopping:** Pipeline rigorosa con validazione ad ogni epoca e salvataggio del modello migliore.
- **Sperimentazione:** Facilmente configurabile tramite file di configurazione e script dedicati.
- **Bilanciamento delle classi:** Possibilità di bilanciare la loss per dataset non perfettamente bilanciati.
- **Salvataggio/caricamento modello:** Gestione automatica dei checkpoint.
- **Quantizzazione e export:** Strumenti dedicati per deployment su diversi dispositivi.

---

## Membri del gruppo

- Jacopo Maria Spitaleri
- Alessandro Dominici
- Seck Mactar Ibrahima
