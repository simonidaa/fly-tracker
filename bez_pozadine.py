from imutils import contours
from skimage import measure
import argparse
import imutils
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from PIL import Image, ImageDraw
import hungarian
import data_storage


input_file = 'C:/Users/marko/Downloads/SSO/Social interaction.mp4'
show_process = False

#izdvajanje pozadine od muva
def get_background(file_path):
    cap = cv2.VideoCapture(file_path)
    # random izvlacenje 200 frejmova i racunanje sredine 
    frame_indices = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=200)
    # cuvanje frejmova u niz
    frames = []
    for idx in frame_indices:
        # frame id postaje bas taj frejm
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        frames.append(frame)
    # racunanje svrednje vrednosti
    median_frame = np.median(frames, axis=0).astype(np.uint8)

    return median_frame

def distance(c1,c2):
    return math.sqrt(pow((c1[0]-c2[0]),2)+pow((c1[1]-c2[1]),2))


cap = cv2.VideoCapture(input_file)
# visina i sirina video frejma
frame_width = int(cap.get(3)*3/4)
frame_height = int(cap.get(4)*3/4)
save_name = f"outputs/{input_file.split('/')[-1]}"
# definisanje i kreiranje VideoWriter objekta
out = cv2.VideoWriter(
    save_name,
    cv2.VideoWriter_fourcc(*'mp4v'), 10,
    (frame_width, frame_height)
)

# vracamo pozadinu snimka koji ucitavamo
background = get_background(input_file)
# konvertovanje pozadine u grayscale format 
background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
# plotovanje pozadine, kesiranje nije najbolje ne raditi ako ne mora
if show_process:
    plt.figure(figsize=(15,15))
    plt.imshow(background, cmap = "gray")
    plt.show()

frame_count = 0
consecutive_frame = 20
#cuvanje starih kordinata
flies_prev = {}
flies_current = {}
fly_count = 0
interaction_count = {}
interaction_dur = 1/int(cap.get(cv2.CAP_PROP_FPS))
#ovo cemo popraviti
print('Pomnoziti ocitane brojeve sa:', interaction_dur)

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame_count += 1

        # konvertovanje frejma u grayscale 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # razlika trenutnog i baznog frejma 
        frame_diff = np.invert(cv2.absdiff(gray, background))

        kernel = np.ones((2, 2), np.uint8)
        # tresholdovanje da prebacimo frejm u binarni
        thresh = ~cv2.threshold(frame_diff, 180, 255, cv2.THRESH_BINARY)[1]
        #img_erosion = ~cv2.erode(~thresh, kernel, iterations=5)
        #thresh = ~(cv2.dilate(img_erosion, kernel, iterations=5))
        thresh = cv2.erode(thresh, kernel, iterations = 1)
        thresh = thresh.astype(np.uint8)

        labels = measure.label(thresh, background=0)
        mask = np.zeros(thresh.shape, dtype="uint8")
        for label in np.unique(labels):
            # ako se labela nalazi na pozadini ignorise se
            if label == 0:
                continue
            # racuna se povrsina tog segmenta kome pripada labela
            labelMask = np.zeros(thresh.shape, dtype="uint8")
            labelMask[labels == label] = 255
            numPixels = cv2.countNonZero(labelMask)
            # ako je blob veci ili manji od nekih zadatih vrednosti to nisu muve
            if numPixels > 20 and numPixels < 60:
                mask = cv2.add(mask, labelMask)

        # pojedi one najmanje linije koje su ostale
        mask = cv2.erode(mask, kernel, iterations = 1)

        # pronadji sve konture i sortiraj ih
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = contours.sort_contours(cnts)[0]

        brojac = 0
        #brojac tocaka za konkretnu iteraciju
        fly_detected = 0
        # idi kroz sve konture
        for (i, c) in enumerate(cnts):
            if(cv2.contourArea(c) > 1):
                # crtaj okvir kod detektovanih muva
                (x, y, w, h) = cv2.boundingRect(c)
                flies_current[brojac] = (x,y)
                #u prvom frejmu prebroj sve muve, stavi im interakcije na 0
                if frame_count == 1:
                    flies_prev[brojac] = flies_current[brojac]
                    interaction_count[fly_count] = 0
                    fly_count += 1
                else: fly_detected += 1
                brojac += 1
        if frame_count == 1:
            continue

        #ne mogu algoritamski ispravljati takve greske kao stu si vise tacaka
        if fly_detected > fly_count: continue
        #popunjavamo nulama jer ako nam fale podaci svjdn cemo dodati optimalne tacke tj. nule za promene
        distance_change = np.zeros((fly_count, fly_count))
        #o(n^2) racunamo distance za svake dve tocke
        for i in range(fly_detected):
            for j in range(fly_count):
                distance_change[i][j] = distance(flies_current[i], flies_prev[j])
        #izdvajamo optimalne indekse
        index_flies_current = hungarian.hungarian_algorithm(distance_change)
        #dodeljujemo nove koordinate optimalnim musicama, ostale racunamo kao izgubljene i nepomicne
        for i, j in index_flies_current:
            #samo ako je detekcija postojala pridruzi je, ako nije ostavi isto
            if i < fly_detected: flies_prev[j] = flies_current[i]
        #bitno nam je da se musica vidja s musicama, ne i velicina bleje, ko smo mi da sudimo
        #moze da se optimizuje vrv nekim pokazivacima al nisam strucan za piton, ostavljam mayi da se igra
        for i in range(fly_count):
            for j in range(i+1, fly_count):
                if distance(flies_prev[i],flies_prev[j]) < 20:
                    interaction_count[i] += 1
                    interaction_count[j] += 1
                    break

        # ubaci u csv
        data_storage.input_trajectory(flies_prev)
        data_storage.input_interactions(interaction_count)


        # prikaz finalne slike, ukoliko je potrebno
        if show_process:
            for f in range(fly_count):
                cv2.putText(frame, "#{}".format(f), (flies_prev[f][0], flies_prev[f][1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            imS = cv2.resize(frame, (frame_width, frame_height))
            cv2.imshow("Image", imS)
            cv2.waitKey(1)
    else:
        break

cap.release()
cv2.destroyAllWindows()
