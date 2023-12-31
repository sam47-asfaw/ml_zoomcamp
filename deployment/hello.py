from flask import Flask

app = Flask('hello')

@app.rout('/hello', methods=['GET'])
def hello_world():
    return "Hello, World"

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port= 8080)