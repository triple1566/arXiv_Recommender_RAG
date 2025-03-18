from flask import Flask, jsonify
from flask_restful import Resource, Api, reqparse, fields, marshal_with, abort
from RAGmodel import df, model_encoder, qdrant_client, initialize_arxiv_data, search_arxiv_papers
import asyncio

DEBUG=False


# Load the LLM Model and initialize the arXiv data
sample_size=5000
print("encoding device: "+str(model_encoder.device))
path, df, model_encoder, qdrant_client, model_loaded = initialize_arxiv_data(df, model_encoder, qdrant_client, sample_size)
#Configure flask app
app=Flask(__name__)
api=Api(app)


arXivPaper = [{
    'id':fields.Integer,
    'title':fields.String,
    'authors':fields.String,
    'abstract':fields.String
}]

#This class inherits from flask_restful's abstract RESOURCE class
class LLMSearch(Resource):
    @marshal_with(arXivPaper)
    def get(self):
        user_args = reqparse.RequestParser()
        user_args.add_argument('prompt', type=str, required=True, help="User prompt is required")
        user_args.add_argument('limit', type=int, default=5, help="Limit must be an integer")
        args=user_args.parse_args()
        userPrompt = args['prompt']
        print("Given " + userPrompt)
        limit = args['limit']
        print("Given " + str(limit))
        #create reponse based on user prompt
        if model_loaded:
            hits = search_arxiv_papers(model_encoder=model_encoder, qdrant_client=qdrant_client, user_prompt=userPrompt, limit_search_to=limit)
            #load response into a list
            result=[]
            for hit in hits:
                paper={
                    'id':fields.Integer,
                    'title':fields.String,
                    'authors':fields.String,
                    'abstract':fields.String,
                }
                paper['id']=hit["id"]
                paper['title']=hit["title"]
                paper['authors']=hit["authors"]
                paper['abstract']=hit["abstract"]
                result.append(paper)
            if len(result)==0:
                abort(404,"Oops, something went wrong!")
            for paper in result:
                print(paper)
            api_response = jsonify(result)
            print(api_response)
            return api_response, 200
        else:
            return jsonify({
                    "message" : "Model not loaded",
                    "is_model_loaded" : model_loaded
                }), 200

#add our LLMSearch class to our api entry point
api.add_resource(LLMSearch, '/api/generateresponse')

@app.route('/')
def index():
    return "<h1>Flask REST Api for Leo's arXiv RAG</h1>"

if __name__=="__main__":
    app.run(debug=DEBUG, use_reloader=False)
