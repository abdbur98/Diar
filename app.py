from SphereDiar import *
import json
from json import JSONEncoder
from flask import Flask ,request
from flask_restful import Api , Resource
import os


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def startDiarization(audioPath , modelPath):
    SD = SphereDiar(audioPath , modelPath)
    x = SD.start()
    numpyData = {"array": x}
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)
    return encodedNumpyData




app = Flask(__name__)
api = Api(app)


UPLOAD_DIRECTORY = "uploadedAudios"


if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

class Temp(Resource):
    def get(self):
        return "HELLO THERE!"
    def post(self):
        pathToAudio = os.path.join(UPLOAD_DIRECTORY, request.files['file'].filename)
        with open(pathToAudio, "wb") as fp:
            fp.write(request.files['file'].read())
        # request.files['file'].save(UPLOAD_DIRECTORY)
        # print(request.files['file'].filename)
        # Return 201 CREATED
        result = startDiarization(pathToAudio,"SphereSpeaker.hdf" )
        os.remove(pathToAudio)
        return result


api.add_resource(Temp , '/')

if __name__ == "__main__":
    app.run(host='0.0.0.0')
