from flaskapp import flask_app
from flaskapp.registerblueprint import registerBluePrint

registerBluePrint(flask_app)
flask_app.run(debug=True)
