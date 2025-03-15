from flask import Flask, jsonify
from flask_restful import Resource, Api, reqparse, fields, marshal_with, abort
from RAGmodel import df, model_encoder, qdrant_client, initialize_arxiv_data, search_arxiv_papers

DEBUG=True


# Load the LLM Model and initialize the arXiv data
sample_size=5000
path, df, model_encoder, qdrant_client, model_loaded = initialize_arxiv_data(df, model_encoder, qdrant_client, sample_size)

#Configure flask app
app=Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///database.db'
api=Api(app)

user_args = reqparse.RequestParser()
user_args.add_argument('prompt', type=str, required=True, help="User prompt is required")
user_args.add_argument('limit', type=int, required=True, help="Limit is required")


arXivPaper = {
    'id':fields.Integer,
    'title':fields.String,
    'authors':fields.String,
    'abstract':fields.String
}

#This class inherits from flask_restful's abstract RESOURCE class
class LLMSearch(Resource):
    @marshal_with(arXivPaper)
    def get(self):
        args=user_args.parse_args()
        userPrompt = args['prompt']
        limit = args['limit']
        #create reponse based on user prompt
        if model_loaded:
            hits = search_arxiv_papers(user_prompt=userPrompt, limit_search_to=limit)
            #load response into a list
            result=[]
            for hit in hits:
                paper={
                    'id':fields.Integer,
                    'title':fields.String,
                    'authors':fields.String,
                    'abstract':fields.String,
                }
                paper['id']=hit.payload["id"]
                paper['title']=hit.payload["title"]
                paper['authors']=hit.payload["authors"]
                paper['abstract']=hit.payload["abstract"]
                result.append(paper)

            if len(result)==0:
                abort(404,"Oops, something went wrong!")
            return jsonify(result), 200
        else:
            return jsonify("Model not loaded"), 500

#add our LLMSearch class to our api entry point
api.add_resource(LLMSearch, '/api/generateresponse')

@app.route('/')
def index():
    return '<h1>Flask REST Api for Leos arXiv RAG</h1>'

if __name__=="__main__":
    app.run(debug=DEBUG)
