from flask import Flask
from flask_restful import Api, Resource
from flask_swagger_ui import get_swaggerui_blueprint

app = Flask(__name__)

# Create the API object
api = Api(app)

class HelloWorld(Resource):
    def get(self):
        """
        Get a simple hello world message
        ---
        responses:
          200:
            description: Success
        """
        return {'message': 'Hello, World!'}

# Add the HelloWorld resource to the API
api.add_resource(HelloWorld, '/hello/world')

# Configure Swagger UI
swagger_ui_blueprint = get_swaggerui_blueprint(
    '/swagger',
    '/static/swagger.json',
    config={
        'app_name': 'Hello World API'
    }
)

# Register the blueprint
app.register_blueprint(swagger_ui_blueprint, url_prefix='/swagger')
