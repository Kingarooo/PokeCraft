import json
import os


#read pokedex json file
pokedex = json.loads(open("pokedex.json").read())

#read types json file
types = json.loads(open("types.json").read())

#create inside data/processed different folders for each type
for type in types:
  if not os.path.exists(f"data/processed/{type['english']}"):
    os.mkdir(f"data/processed/{type['english']}")

#copy each image in processed/all to each folder of his type
for file in os.listdir("data/processed/all"):
  pokemon_id = file.split(".")[0].split("-")[1]
  type = pokedex[int(pokemon_id)]["type"][0]
  os.system(f"cp data/processed/all/{file} data/processed/{type}/{file}")
