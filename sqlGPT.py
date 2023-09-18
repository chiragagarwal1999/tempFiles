model_name = "NumbersStation/nsql-350M"

from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

table_schema = """CREATE TABLE "t_travel_track" (
"TrackId" INTEGER NOT NULL,
"Name" NVARCHAR(200) NOT NULL,
"AlbumId" INTEGER,
"MediaTypeId" INTEGER NOT NULL,
"GenreId" INTEGER,
"Composer" NVARCHAR(220),
"Milliseconds" INTEGER NOT NULL,
"Bytes" INTEGER,
"UnitPrice" NUMERIC(10, 2) NOT NULL,
PRIMARY KEY ("TrackId"),
FOREIGN KEY("MediaTypeId") REFERENCES "MediaType" ("MediaTypeId"),
FOREIGN KEY("GenreId") REFERENCES "Genre" ("GenreId"),
FOREIGN KEY("AlbumId") REFERENCES "Album" ("AlbumId")
)
"""

question = "How many tracks are composed by Arijit Singh and are priced above 25 ?"

prompt = f"""{table_schema}

-- Using valid SQLite, answer the following questions for the tables provided above.

-- {question}

SELECT"""

input_ids = tokenizer(prompt, return_tensors="pt").input_ids
generated_ids = model.generate(input_ids, max_length=500)
output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
output = 'SELECT' + output.split('SELECT')[-1]

print(output)
