from flask import Flask
from flask_restplus import Api, Resource

app = Flask(__name__)
api = Api(
    app,
    version='1.0',
    title='Hello World API',
    description='A simple Hello World API',
    doc='/swagger'
)

ns = api.namespace('hello', description='Hello World operations')

@ns.route('/world')
class HelloWorld(Resource):
    @ns.doc('get_hello_world')
    @ns.response(200, 'Success')
    def get(self):
        """Return a simple hello world message."""
        return {'message': 'Hello, World!'}

#if __name__ == '__main__':
#    app.run(debug=True)