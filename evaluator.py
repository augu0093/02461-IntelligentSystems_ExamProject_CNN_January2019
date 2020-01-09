from keras.models import load_model
from tkinter.filedialog import askopenfilename, askdirectory
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.metrics import accuracy_score

def __get_image_generator():
    test_datagen = ImageDataGenerator()
    test_set = test_datagen.flow_from_directory('images/test',
                                 batch_size=50,
                                 target_size=(200,200))
 
    return test_set

model_path = askopenfilename()
model = load_model(model_path)
generator = __get_image_generator()

count = 0
acc = []
for i in range(30):
    loss, a = model.evaluate_generator(generator, steps=25)
    acc.append(a)
    count += 1
    print(count)

df = pd.DataFrame({'val_acc': acc})
df.to_csv('val_acc_best.csv')