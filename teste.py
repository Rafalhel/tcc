import cv2, numpy as np
# img1 = None
# img1 = cv2.imread('foto.jpeg')
img1 = cv2.imread('foto1.jpg')

# img = cv2.cvtColor(cv2.imread(f"imagens\imagemTeste.jpg"), cv2.COLOR_BGR2GRAY).flatten()
# img = cv2.imread("imagens\imagemTeste.jpg", cv2.IMREAD_GRAYSCALE)
# img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# img = cv2.cvtColor(cv2.imread(f"imagens\imagemTeste.jpg"), cv2.COLOR_BGR2GRAY).flatten()
# dataFrame = pd.DataFrame(img)
win_name = 'Camera Matching'
MIN_MATCH = 10
# ORB Geração de detectores  ---①
detector = cv2.ORB_create(1000)
# Flann Criar extrator ---②
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6,
                   key_size = 12,
                   multi_probe_level = 1)
search_params=dict(checks=32)
matcher = cv2.FlannBasedMatcher(index_params, search_params)
# Vincule a captura da câmera e reduza o tamanho do quadro ---③
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():

    ret, frame = cap.read()
    if img1 is None:  # Nenhuma imagem registrada, desvio da câmera
        res = frame
    else:
        try:# Se houver uma imagem registrada, a correspondência começa
            img2 = frame
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            # Extraia pontos-chave e descritores
            kp1, desc1 = detector.detectAndCompute(gray1, None)
            kp2, desc2 = detector.detectAndCompute(gray2, None)
            tOrFalse = np.any(desc2)

            if tOrFalse == None:
                res = frame
            else:
            # com k=2 knnMatch

                matches = matcher.knnMatch(desc1, desc2, 2)
                # Extração de bons pontos de correspondência com 75% da distância do vizinho---②
                ratio = 0.75
                good_matches = [m[0] for m in matches \
                                    if len(m) == 2 and m[0].distance < m[1].distance * ratio]
                print('good matches:%d/%d' %(len(good_matches),len(matches)))
                # Preencha a máscara com zeros para evitar desenhar todos os pontos correspondentes
                matchesMask = np.zeros(len(good_matches)).tolist()
                # Se for mais do que o número mínimo de bons pontos de correspondência
                if len(good_matches) > MIN_MATCH:
                    # Encontrar as coordenadas das imagens de origem e destino com bons pontos de correspondência ---③
                    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ])
                    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ])
                    # Encontre a Matriz de Transformação de Perspectiva ---⑤
                    mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    accuracy=float(mask.sum()) / mask.size
                    print("accuracy: %d/%d(%.2f%%)"% (mask.sum(), mask.size, accuracy))
                    if mask.sum() > MIN_MATCH:  # No caso de mais do que o número mínimo de pontos de correspondência normal
                        # Defina a máscara para desenhar apenas os pontos de correspondência discrepantes
                        matchesMask = mask.ravel().tolist()
                        # Área de exibição após a transformação da perspectiva para as coordenadas da imagem original  ---⑦
                        h,w, = img1.shape[:2]
                        pts = np.float32([ [[0,0]],[[0,h-1]],[[w-1,h-1]],[[w-1,0]] ])
                        dst = cv2.perspectiveTransform(pts,mtrx)
                        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
                # Desenhe pontos correspondentes com uma máscara ---⑨
                res = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, \
                                    matchesMask=matchesMask,
                                    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

        except:
            res = frame

    # emita o resultado

    ### Mostra imagem e os pontos iguais
    # cv2.imshow(win_name, res)
    ### Não moptra e pontos iguais
    cv2.imshow(win_name, frame)
    key = cv2.waitKey(1)
    if key == 27:    # Esc, fim
            break
    elif key == ord(' '): # Defina o ROI com a barra de espaço para definir img1
        x,y,w,h = cv2.selectROI(win_name, frame, False)
        if w and h:
            img1 = frame[y:y+h, x:x+w]
else:
    print("can't open camera.")
cap.release()
cv2.destroyAllWindows()