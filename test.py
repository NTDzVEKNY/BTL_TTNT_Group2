import cv2
import numpy as np
import keras
import jax
from keras.models import load_model

cap = cv2.VideoCapture(0)

# Load model đã train
my_model = load_model("vggmodel.keras")
my_model.load_weights("weights-41-0.90.weights.h5")

while (True):

    ret, image_org = cap.read()
    if not ret:
        continue
    image_org = cv2.resize(image_org, dsize=None, fx=0.5, fy=0.5)

    image = image_org.copy()
    image = cv2.resize(image, dsize=(128, 128))
    image = image.astype('float') * 1. / 255
    # Chuyển thành tensor
    image = np.expand_dims(image, axis=0)

    # Predict
    predict = my_model.predict(image)
    prediction = predict[0][0]

    if prediction >= 0.5:
        label = "dog"
    else:
        label = "cat"

    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (120, 50)
    fontScale = 1.5
    color = (0, 255, 0)
    thickness = 2

    cv2.putText(image_org, label, org, font, fontScale, color, thickness, cv2.LINE_AA)
    print(predict[0])
    print(predict[0][0])

    cv2.imshow("Picture", image_org)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
