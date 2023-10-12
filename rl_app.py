from flask import Flask, request, jsonify
from related_links import related_links

app = Flask(__name__)

@app.route('/related-urls', methods=['POST'])
def get_related_urls():
    data = request.get_json()
    url = data['url']
    related_urls = related_links(url)
    return jsonify(related_urls)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)