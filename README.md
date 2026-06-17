# facial-expression-recognition
Facial Expression Recognition — FER2013 (PyTorch + Wandb)

Kaggle-ის Challenges in Representation Learning: Facial Expression Recognition Challenge (FER2013).

ამ პროექტის მიზანი იყო დეტალურად დოკუმენტირებული, ეტაპობრივი კვლევის ჩატარება: პატარა baseline მოდელიდან დაწყება, შემდეგ არქიტექტურის თანდათან გაზრდა, ყოველ ეტაპზე underfitting-ისა და overfitting-ის ანალიზი, თითოეული არქიტექტურისთვის hyperparameter tuning-ის ჩატარება და class imbalance-ის პრობლემასთან გამკლავება. ყველა ექსპერიმენტი წარმოდგენილი იყო ფორმატში hypothesis → result → interpretation და ყველაფერი იწერებოდა Weights & Biases-ში.

საუკეთესო მოდელი: tuned SmallResNet + weighted sampler — test accuracy 0.681, macro-F1 0.677
(ადამიანის საშუალო შედეგი FER2013-ზე დაახლოებით 65%-ია).

Wandb project: https://wandb.ai/tasomamaladze123-none/fer2013
Stack: Google Colab (T4 GPU) · Kaggle API (data) · GitHub (code) · Weights & Biases (tracking)

## Repository Structure
facial-expression-recognition/
├── README.md
├── fer_experiments-1.ipynb
├── fer_experiments-2.ipynb
├── src/
│   ├── data.py
│   ├── models.py
│   ├── train.py
│   └── utils.py
data.py შეიცავს Dataset-ს, transforms-ს, imbalance helper-ებს და DataLoader-ებს.
models.py შეიცავს TinyCNN, DeeperCNN და SmallResNet მოდელებს.
train.py პასუხისმგებელია training loop-ზე, Wandb logging-ზე, LR scheduler-ზე და sweep-ებზე.
utils.py შეიცავს seeding-ს და sanity check-ებს.

## Data და EDA

FER2013 შეიცავს 35,887 grayscale 48×48 სურათს და 7 ემოციას.

Dataset დავყავი მკაცრად Usage სვეტის მიხედვით (FER2013-ის სტანდარტული პროტოკოლი, leakage-ის გარეშე):

Training → train (28,709)
PublicTest → validation (3,589)
PrivateTest → test (3,589)
მთავარი პრობლემა — class imbalance

Train set-ში კლასების რაოდენობები:

Angry: 4,953
Disgust: 547
Fear: 5,121
Happy: 8,989
Sad: 6,077
Surprise: 4,002
Neutral: 6,198

Disgust დაახლოებით 16-ჯერ უფრო იშვიათია, ვიდრე Happy.

Test set-ში Disgust-ს მხოლოდ 55 სურათი აქვს. შედარებისთვის, შემდეგი ყველაზე პატარა კლასი (Surprise) უკვე 416 სურათს შეიცავს.

ეს ნიშნავს, რომ Disgust recall საკმაოდ არასტაბილური მეტრიკაა. თითოეული სწორად კლასიფიცირებული სურათი დაახლოებით 1.8%-ს ცვლის.

ამ მიზეზით accuracy-სთან ერთად ყოველთვის ვრეპორტავდი macro-F1-საც და ცალკე imbalance ექსპერიმენტებიც ჩავატარე.

Pixels ერთხელ გადაიპარსა uint8 array-ში (~82 MB RAM).
Normalization გამოვითვალე უშუალოდ train set-ზე:
mean ≈ 0.507
std ≈ 0.255
Augmentation (ჩასართავი/გამოსართავი):
horizontal flip
±10° rotation
მცირე translation
vertical flip არ გამოვიყენე, რადგან თავდაყირა სახეები რეალურ distribution-ს არ ეკუთვნის.

## Forward/Backward Sanity Checks

Forward Check
untrained 7-კლასიანი classifier-ის cross-entropy თეორიულად უნდა იყოს:
ln(7) = 1.946
მიღებული შედეგი:
1.945
ეს ნიშნავს, რომ labels, output layer და loss function სწორად არის მიერთებული.

Backward Check
მოდელს ერთი batch უნდა დაემახსოვრებინა სრულყოფილად.
შედეგი:
train accuracy = 1.000
loss = 0.0003
ანუ gradients სწორად ვრცელდება მთელ ქსელში.

ასევე ვიყენებდი controlled comparisons-ს:

ჯერ ყველა მოდელი იდენტური hyperparameter-ებით გავუშვი, რათა მხოლოდ არქიტექტურის ეფექტი დამენახა, შემდეგ კი თითო ჯერზე მხოლოდ ერთ პარამეტრს ვცვლიდი.

## Architectures

ყველა baseline მოდელი გაშვებული იყო ერთნაირი პარამეტრებით:
augment=False
weight decay=0
lr=1e-3
batch size=64
optimizer=Adam
epochs=20

ასე არქიტექტურის გავლენა მაქსიმალურად იზოლირებული რჩებოდა.

Architecture	Test Accuracy	Test F1
TinyCNN	0.514	0.486
DeeperCNN	0.670	0.662
SmallResNet	0.598	0.514

## Hypothesis → Result → Interpretation:

ჰიპოთეზა: რაც უფრო ღრმაა მოდელი, მით უკეთესი შედეგი უნდა მქონოდა.
შედეგი: ასე არ მოხდა — regularization-ის გარეშე SmallResNet-მა (0.598) უარესი შედეგი აჩვენა, ვიდრე საშუალო ზომის DeeperCNN-მა (0.670).
ინტერპრეტაცია: capacity შესაბამისი regularization-ის გარეშე generalization-ს აუარესებს.
TinyCNN overfit-ს აკეთებს მიუხედავად იმისა, რომ პატარაა. მისი 37k პარამეტრიდან ~32k ერთ regularization-ის გარეშე დარჩენილ FC layer-შია (4608→7), რომელიც training-ის pixel pattern-ებს იმახსოვრებს, მაშინ როცა conv backbone სუსტია. (თავიდან underfitting ვივარაუდე და ვცდებოდი ცოტა პარამეტრი ყოველთვის არ ნიშნავს underfitს.)
DeeperCNN არის sweet spot unregularized პარამეტრებისთვის: BN + Dropout gap-ს ზომიერად ინარჩუნებს და baseline-ებიდან საუკეთესო generalization აქვს.
SmallResNet კატასტროფულად overfit-ს აკეთებს (train 0.965, val loss 2.35-მდე აფეთქდა) regularization-ის გარეშე მისი capacity უპირატესობის ნაცვლად ნაკლი ხდება.


Per-class მტკიცებულება - overfitting ყველაზე იშვიათ კლასს ვნებს. Disgust recall: C baseline (severe overfit) = 10.9%, ხოლო B baseline = 70.9%. overfit მოდელი თავის capacity-ს majority კლასების დამახსოვრებაში ხარჯავს და Disgust-ს პრაქტიკულად ტოვებს. ეს ცხადად აჩვენებს, რომ overfitting კლასებზე თანაბრად არ მოქმედებს.

## Regularization Experiments

თითო ცვლადს ცალკე ვცვლიდი, რომ ეფექტი იზოლირებული ყოფილიყო:

Run / Model          Test Acc   Test F1   Disgust Recall
---------------------------------------------------------
A baseline           0.514      0.486     34.5%
A regularized        0.529      0.454     16.4% (↓)
C baseline (overfit) 0.598      0.514     10.9%
C regularized        0.656      0.621     58.2% (↑)


A regularized: overfitting აღმოიფხვრა (val loss აღარ იზრდება; train ახლა val-ზე დაბლაა, რადგან augmentation + აქტიური Dropout training pass-ს უფრო ართულებს, ვიდრე evaluation-ს). accuracy ოდნავ გაიზარდა, მაგრამ macro-F1 დაეცა — per-class რიცხვები ხსნის რატომ: Disgust recall 34.5% → 16.4% დაეცა. სუსტ TinyCNN backbone-ზე augmentation-მა მოდელი majority კლასებისკენ წაიყვანა; მთლიანი მოგება მხოლოდ majority კლასებზე იყო.
C regularized: ~37 pt gap ~4 pt-მდე შემცირდა, test 0.598 → 0.656, და Disgust recall 10.9% → 58.2% აღდგა, majority კლასების დამახსოვრების აღკვეთამ მოდელს იშვიათი კლასის სწავლის საშუალება მისცა. თუმცა flat LR-ითა და 30 epoch-ით under-trained დარჩა (train ჯერ კიდევ იზრდებოდა), ამიტომ ჯერ B-ს ვერ გაუსწრო.


## Hyperparameter Sweeps (Wandb Sweeps)

თითო არქიტექტურისთვის ჩავატარე Bayesian sweep (15-epoch runs სწრაფი შედარებისთვის) lr / optimizer / batch / weight decay / dropout / scheduler-ზე, optimization metric - val/acc.

მთავარი დასკვნა ორივე sweep-ში: Adam run-მა ყველა SGD run-ს აჯობა; SGD დაბალ lr-ზე cosine-ით 15 epoch-ში ძლიერად under-train-ს აკეთებდა.

Sweep / Winning Config                        Best Val
------------------------------------------------------
SmallResNet (Adam, lr=1.2e-3, bs=64,
dropout=0.2, cosine, wd=5e-4)                 0.639

DeeperCNN (Adam, lr=1.9e-3, bs=64,
dropout=0.3, cosine, wd=1e-4)                 0.642

საბოლოო tuned SmallResNet (გამარჯვებული config, cosine 45 epoch-ზე): train 0.749 / best val 0.673 / test 0.677 / F1 0.647, gap ~7.6 pt, val loss ~0.95-ზე დაconvergeდა. სწორი LR schedule-ითა და საკმარისი epoch-ებით SmallResNet-მა საბოლოოდ მიაღწია თავის უფრო მაღალ ceiling-ს და DeeperCNN-ს გაუსწრო, რაც ადასტურებს ჰიპოთეზას, რომ მისი დაბალი baseline შედეგი regularization/training-ის და არა capacity-ის პრობლემა იყო.

## Class Imbalance Experiments

tuned SmallResNet config-ზე (30 epoch), strategy-ების შედარება:

Strategy                           Test Acc   Macro-F1   Disgust Recall
-------------------------------------------------------------------------
Unweighted                        0.665      0.626      58.2%
Weighted loss (inverse-freq)      0.640      0.617      —
Weighted sampler (oversampling)   0.681      0.677      72.7%

ჰიპოთეზა: imbalance handling minority კლასის შედეგს უნდა აუმჯობესებდეს.
Weighted loss-მა პირიქით იმუშავა. inverse-frequency weighting Disgust-ს ~9.4× წონას აძლევს, რაც დიდ, არასტაბილურ gradient-ებს ქმნის პატარა (547 მაგალითი) და ხმაურიან კლასზე. training დესტაბილიზდა და ყველაზე ნელა მოახდინა convergence (train მხოლოდ 0.657 30 epoch-ზე); ორივე metric დაეცა. მექანიზმი: loss-weighting ყველა Disgust მაგალითის gradient-ს აძლიერებს, მათ შორის არასწორად მონიშნულებს (FER2013 ცნობილია label noise-ით), ამიტომ მოდელი ნაწილობრივ noise-ს მისდევს.
Weighted sampler-მა გაიმარჯვა. oversampling data distribution-ს აბალანსებს per-example gradient-ის გაზრდის გარეშე ბევრად რბილია. training სტაბილური დარჩა და ყველაფერი გააუმჯობესა: accuracy, macro-F1 (+5 pt) და Disgust recall 72.7%-მდე (precision კვლავ მაღალი — 81.6%, ანუ false-positive-ების ნაკადი არ შექმნა).


## საუკეთესო მოდელის სრული per-class breakdown:

Class (Train / Test)        Recall     Precision     F1
--------------------------------------------------------
Happy    (8989 / 879)       86.8%      90.5%        0.89
Surprise (4002 / 416)       81.5%      75.2%        0.78
Disgust  (547 / 55)         72.7%      81.6%        0.77
Neutral  (6198 / 626)       68.4%      65.8%        0.67
Angry    (4953 / 491)       63.3%      57.9%        0.61
Sad      (6077 / 594)       55.2%      52.1%        0.54
Fear     (5121 / 528)       44.3%      54.5%        0.49

oversampling-ის შემდეგ ყველაზე იშვიათი კლასი (Disgust) მე-3 საუკეთესო კლასია, Neutral, Angry, Sad და Fear-ზე მაღლა, რომელთაც 8–11× მეტი data აქვთ. bottleneck აღარ არის class frequency, არამედ ემოციების შინაგანი ბუნდოვანება: მთავარი შეცდომებია Fear→Sad (103), Sad→Neutral (101), Fear→Angry (85), Angry→Sad (78), ვიზუალურად მსგავსი negative ემოციები.

## Results Summary

Run / Arch / Settings                         Acc      Macro-F1   Disgust Recall
---------------------------------------------------------------------------------
A baseline TinyCNN (no reg)                   0.514    0.486      34.5%
A regularized TinyCNN (aug + wd)              0.529    0.454      16.4%
B baseline DeeperCNN (no reg)                 0.670    0.662      70.9%
C baseline SmallResNet (no reg)               0.598    0.514      10.9%
C regularized SmallResNet (aug + wd, 30 ep)   0.656    0.621      58.2%
C final (tuned) SmallResNet (sweep, cosine)   0.677    0.647      —
C + weighted loss SmallResNet (imbalance)     0.640    0.617      —
C + weighted sampler SmallResNet (best)        0.681    0.677      72.7%
