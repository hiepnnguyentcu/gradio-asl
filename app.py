import gradio as gr
import numpy as np
import mediapipe as mp
import tflite_runtime.interpreter as tflite
import cv2
import shutil
import os

def flatten_coords(lmarks):
    landmarks = []
    for pos in lmarks.landmark:
        landmarks.append([pos.x,pos.y,pos.z])
    return landmarks

char_to_idx = {" ":0,"!":1,"#":2,"$":3,"%":4,"&":5,"'":6,"(":7,")":8,"*":9,"+":10,",":11,"-":12,".":13,"/":14,"0":15,"1":16,"2":17,"3":18,"4":19,"5":20,"6":21,"7":22,"8":23,"9":24,":":25,";":26,"=":27,"?":28,"@":29,"[":30,"_":31,"a":32,"b":33,"c":34,"d":35,"e":36,"f":37,"g":38,"h":39,"i":40,"j":41,"k":42,"l":43,"m":44,"n":45,"o":46,"p":47,"q":48,"r":49,"s":50,"t":51,"u":52,"v":53,"w":54,"x":55,"y":56,"z":57,"~":58}
idx_to_char = {v:k for k,v in char_to_idx.items()}

class InferenceModule:
    def __init__(self,idx_to_char=idx_to_char,model_path="./weights/model.tflite",bg_color=(0, 0, 0),debug=True):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.interpreter = tflite.Interpreter(model_path)
        self.BG_COLOR = bg_color
        self.debug=debug
        self.prediction_fn = self.interpreter.get_signature_runner("serving_default")
        self.idx_to_char = idx_to_char
        if self.debug:
            os.makedirs("./tmp",exist_ok=True)
        
        #Test if the model works.
        output = self.prediction_fn(inputs=np.random.rand(1, 164).astype(np.float32))
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        print("Input Shape:", input_details[0]['shape'])
        print("Output Shape:", output_details[0]['shape'])
        print("successfully setup model for inference")
    def __call__(self,video_path):
        if not video_path:
            return "Prediction: NaN"
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS) # Get the frames per second of the video
        all_landmarks = []
        save_frames =[]
        if self.debug:
            shutil.copy(video_path,"./tmp")
        with self.mp_holistic.Holistic(
            static_image_mode=False, # Set to False for video.
            model_complexity=2,
            enable_segmentation=True,
            refine_face_landmarks=False) as holistic:

            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break
                results = holistic.process(frame)

                face = flatten_coords(results.face_landmarks) if results.face_landmarks else [[np.nan]*3 for _ in range(468)]
                pose = flatten_coords(results.pose_landmarks) if results.pose_landmarks else [[np.nan]*3 for _ in range(33)]
                lefthand = flatten_coords(results.left_hand_landmarks) if results.left_hand_landmarks else [[np.nan]*3 for _ in range(21)]
                righthand = flatten_coords(results.right_hand_landmarks) if results.right_hand_landmarks else [[np.nan]*3 for _ in range(21)]

                frame_landmarks = face+lefthand+pose+righthand
                all_landmarks.append(frame_landmarks)
                if self.debug:
                    annotated_image = frame.copy()
                    self.mp_drawing.draw_landmarks(
                        annotated_image,
                        results.pose_landmarks,
                        self.mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1))
                    self.mp_drawing.draw_landmarks(
                        annotated_image,
                        results.left_hand_landmarks,
                        self.mp_holistic.HAND_CONNECTIONS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1))
                    self.mp_drawing.draw_landmarks(
                        annotated_image,
                        results.right_hand_landmarks,
                        self.mp_holistic.HAND_CONNECTIONS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1))
                    save_frames.append(annotated_image)
        if self.debug:
            out = cv2.VideoWriter('./tmp/with_landmarks.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (annotated_image.shape[1],annotated_image.shape[0]))
            for frame in save_frames:
                out.write(frame)
            out.release()
        input_data = np.array(all_landmarks, dtype=np.float32)
        output = self.prediction_fn(inputs=input_data.reshape(1,-1,1629))
        text = "".join(self.idx_to_char[int(x)] for x in output["outputs"].argmax(1))
        if self.debug:
            print("predicted text is: ",text)
        return f"prediction: {text}"


infer = InferenceModule(debug=True)


webcam = gr.Interface(
    fn=infer, 
    inputs=[gr.Video(label="Webcam",format="mp4")],
    outputs= "text", 
    live=False,
    concurrency_limit=10,
    title="American Sign Language Fingerspelling Prediction from Video")
""" 
upload = gr.Interface(
    fn=infer, 
    inputs=[gr.Video(source="upload", label="Upload",format="mp4",optional=True)], 
    outputs= "text", 
    live=False,
    enable_queue=True,
    title="American Sign Language Fingerspelling Prediction from Video")
"""
demo = gr.TabbedInterface([webcam], ["webcam-input"])

#demo.queue(concurrency_count=2)
demo.launch(share=True)

