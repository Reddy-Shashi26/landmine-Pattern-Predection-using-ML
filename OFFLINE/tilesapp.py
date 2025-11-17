from flask import Flask, send_from_directory

app = Flask(__name__)

@app.route('/<path:filename>')
def serve_tiles(filename):
    return send_from_directory('tiles', filename)  # 'tiles' is the folder containing your map tiles

if __name__ == '__main__':
    app.run(port=8080)