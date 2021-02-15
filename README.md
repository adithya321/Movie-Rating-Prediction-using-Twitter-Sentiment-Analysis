# Movie Rating Prediction using Twitter Sentiment Analysis

A System to Predict Movie Rating based on Sentiment of tweets about a movie.

- Used **Twitter API** to collect tweets about a movie.
- Used **Python** for **Data cleaning**.
- Used **Naive Bayes Classiﬁer** to classify sentiment of tweets.
- Calculated rating of movie based on sentiment classiﬁcations.

## Run

```bash
cat << EOF > .env
API_KEY='XYZ'
API_SECRET_KEY='XYZ'
BEARER_TOKEN='XYZ'
ACCESS_TOKEN='XYZ'
ACCESS_TOKEN_SECRET='XYZ'
EOF
```

```bash
{
    python3 -m venv venv
    source venv/bin/activate

    pip3 install -r requirements.txt

    python3 tex.py
}
```

```output
First 200 sample tweets
[{'text': 'Watching Thor Ragnarök, i love this movie so much'}, {'text': 'RT @Not_UR_Usual: Thor Ragnarok is the best Thor movie. Debate your mother on this'}, {'text': "@CrackFiend_Fred Dr. Strange is like a phase 1 movie. It's akin to Thor. Civil War is also good, but he was written… https://t.co/9Z8mxuFd5N"}, {'text': '@sithsmcu @joshbattinson Speaking as someone who didn’t hate but didn’t love her movie, Carol never really seems to… https://t.co/IfQli5gDln'}, {'text': 'RT @yonoexiste: in love with her since the first thor movie \nhttps://t.co/enfXQkWkzp'}, {'text': 'in love with her since the first thor movie \nhttps://t.co/enfXQkWkzp'}, {'text': 'just caught myself almost feeling nostalgic for the positive feeling i had about marvel movies before i came to kno… https://t.co/PKPY1YsEln'}, {'text': 'IMA BE MAKING A AVENGERS INFINTY WAR ROLEPLAY IN ROYALE HIGH WE NEED A CAPTAIN AMERICA IRON MAN THOR VISON HULK HAW… https://t.co/IZTqOvnfYB'}, {'text': '@Grizz_NFL But the actual movie itself destroys any marvel movie ... that’s a fact ... are they gonna make Thor hav… https://t.co/nif1GCPhrq'}, {'text': 'Movie script post time, me and trya harp: nah nigg a that niggas real as s name is thor... And Paul in the holy bib… https://t.co/GKGb2KrNxu'}, {'text': 'bitches really want the solo thor movie to feature the guardians as part of the main characters like shut up'}, {'text': 'marvel give me a movie of thor loki valkyrie jane and bruce i dare you i fuckin dare you'}, {'text': '@AdamUnger_7 I was just telling my wife that seeing Endgame in the theater is my favorite movie experience. The ene… https://t.co/Wfw5bCEviR'}, {'text': '@lovethundernews They had potential for a great Thor team up movie by having him with all of the Guardians, but the… https://t.co/ETl6WFcTpA'}, {'text': "@DT2ComicsChat Good selection. \n\nI don't mind Thor in the earlier stuff but once he became more jokey in ragnorok..… https://t.co/ipCquWxnDT"}, {'text': 'Thor: Ragnarok is seriously a GEM of a movie. 😂'}, {'text': '@AlwaysKylo 100% gonna have a 80s movie workout montage with Thor and Quill'}, {'text': '@JC1053 That Thor movie is one of my favs, so funny with the action included. The Rock dude makes that show great as well.'}, {'text': '@_Valentineeeee Yeah they nerfed him after the first avengers movie Thor with stormbreaker and captain marvel can b… https://t.co/Qk8Wsdpoph'}, {'text': 'We captain held Thor’s hammer at my movie theater, everyone went ballistic. It was my favorite moment ever in a mov… https://t.co/xjg3lf393o'}, {'text': "@TR_Multi @realHARTHUR @JaxterTS It's like every DC movie is just Thor the Dark World but Shazam is Ragnarok"}, {'text': 'can we stop acting like ragnarok is the best thor movie, just because y’all can’t deal with more action than humor.'}, {'text': 'Jetzt vorbestellen: Thor: Ragnarok - Stan Lee - Exclusive Movie Masterpiece 1:6 Actionfigur: Preis: 324,00\xa0€ Vorbes… https://t.co/XBpWTMHcnD'}, {'text': 'THOR: RAGNAROK (2017) Movie Clip - God of Thunder | Marvel Studios HD - YouTube https://t.co/Wb6qmdhfq2 これを大画面で見たくてせっせとドリパス投票してるのよー'}, {'text': "@taruyison I'm Stephen is the sorcerer supreme, he meets a bunch of people off earth and is drawn to the unknown, y… https://t.co/gmTqPXRLSI"}, {'text': 'Storm from the X-Men is apparently getting her MCU debut in the next Thor movie. Fan service 👌'}, {'text': '@JazzSwiggity I have many, some of them: shrek, the croods, deadpool, ant man, thor, the last: naruto the movie, sp… https://t.co/kZjEIOXtED'}, {'text': "RT @Socrateaser3: @mrtonylee @Mr_Jim_G @ray8fisher When Marvel was criticized for it's light tone what did they do ? They doubled down on i…"}, {'text': '@torres_criado @_JoseRio97 Parece un scary movie de Thor.\nEs entretenida, pero una vergüenza para la saga.'}, {'text': 'Marvel stans are delusional, they act like Thor was a good movie https://t.co/7V88Vhz23v'}, {'text': "RT @EliteAJITHIANS: [ At Last! After The Long Wait Of\n '481 DAYS' ]\n\n#ThalaAJITH's Upcoming Biggie #Valimai Movie Official Updates Has Been…"}, {'text': 'RT @MysiePereira: The new Guardians of the Galaxy movie looks great :)\n\n#Avengers #AvengersEndgame #endgame #gotg #mcu #marvel #nebula #roc…'}, {'text': "@talkinaboutjane I am expecting the movie being 'We need Thor',  'where is Thor with the Guardians' ..And Thor leav… https://t.co/5PW23hPwf8"}, {'text': "@jotunemo @StarOfAsgard It's hard to balance everyone in such a huge movie though. I think with Val especially Thor… https://t.co/3yI5cHifyJ"}, {'text': '@myghty_thor 同士よ、ありがとう😂 https://t.co/2t5YohYHrg'}, {'text': '@tonygoldmark It just smells of a terrible DC movie. Crazy that the closest Marvel ever got to this was Thor: Dark… https://t.co/n8Ht0LzCfk'}, {'text': 'anyone else have an insane love for the movie ‘thor: raganarok’, cause I know I sure do'}, {'text': 'RT @YoungKawaki: Rewatching Age of Ultron since I barely remember it and seeing Cap move Thor’s hammer when he tried to lift it gave me chi…'}, {'text': 'Who wins this nasty four-way movie character showdown? Magneto, Loki, Doctor Doom or Doctor Sivana?\nLocation: Dubai… https://t.co/7PMmkGdDTv'}, {'text': 'Whoever loves marvel and Thor Mortal the movie was absolutely fucking amazing 🤩 I highly encourage you to sit down and watch it!'}, {'text': '@KingslayerTX Ragnarok broke the mold and allowed Thor, (Chris Hemsworth) to really shine so the writing alone make… https://t.co/tkOzJ5j1O9'}, {'text': '@Timothy67391581 @DanRomens @snydercut in a market where Iron Man passes 1 billion and Thor comes close, a Superman… https://t.co/jNJsNxfefU'}, {'text': '@movie_501 僕も行きたいです🙃\n一緒にいきましょうww'}, {'text': '@NiiteTitan @Pickzit @joshbattinson Imagine if we said that back in the Iron Man 1, Thor ,Cap America and Incredibl… https://t.co/78OFUiET8K'}, {'text': 'Er...how true is the rumour about Anna Diop shooting for the new Thor movie? *if she is, can she be Storm, please???*'}, {'text': 'RT @itsjustanx: Billy and Tommy are having a little too much fun with uncle Cap’s shield and Thor’s hammer. #WandaVision https://t.co/3UQ4M…'}, {'text': '@KCDRisREAL @GodEmperorBoss Difference between subjective and objective. Subjective is having an opinion based off… https://t.co/zd7a92CijZ'}, {'text': "dan howell's favourite marvel movie is Thor"}, {'text': "@GailSimone Malekith. I didn't mind that he was a cookie cutter big bad in Thor: the Dark World because I'd barely… https://t.co/srYpbCjfk6"}, {'text': "@jotunemo yep. I'm also not that keen on seeing the Guardians in a Thor movie actually... but I trust Taika to find a good balance."}, {'text': '@ChadManic wait which really bad thor movie????'}, {'text': 'I mean the movie is called THOR so https://t.co/6QCmMeAj51'}, {'text': 'RT @im_sathiskumar: High time our guys stop asking for update from the ppl who are not involved in the movie. Though it is for fun, it is g…'}, {'text': "@Rajdeeptodcase @SteelbookCaskey @gamps96 @TNeenan Shazam was fun but that's only because I went in expecting somet… https://t.co/h53BA7up78"}, {'text': "@mrtonylee @Mr_Jim_G @ray8fisher When Marvel was criticized for it's light tone what did they do ? They doubled dow… https://t.co/MkwaU8ZbMN"}, {'text': 'When Cap gets Thor’s hammer, goosebumps!!! Also “on your right” and the entire army of previously snapped away good… https://t.co/GpomNStTiA'}, {'text': '@aplacetohide Yeah i am not a big fan of dc movies i love all the marvel movies, watched most of them on the first… https://t.co/n9KDSANUnC'}, {'text': 'RT @ETCanada: #KatDennings also reflects on returning to play Darcy Lewis eight years after the first Thor movie\nhttps://t.co/bGbGbWIsBq'}, {'text': '@MCU_Fanatics @comlcbookfan Most people forget but already exist a thor video game, it was released with the 1st movie'}, {'text': '@kiwicherrylove 1. Doctor Strange \n2. CA: The Winter Solider \n3. Endgame \n4. Thor : Ragnarok \n5. Ant Man ( pls i lo… https://t.co/AxPefQg1V5'}, {'text': '@mrchrisaddison @Baddiel Thing is MCU movies went too far the other way. By the last Thor one it was an out and out… https://t.co/f7Kij6aZUr'}, {'text': 'Remember when this guy tried to take Visions mind stone?\n\nRemember when he told Thanos about the Avengers?\n\nRemembe… https://t.co/8dE8X6ztCc'}, {'text': '@nataliereed84 I think ragnarok was supposed to be a very different hulk movie that ended up being shrunk and put i… https://t.co/agX9s7LlJj'}, {'text': 'Also, side question, Odin was about to crown Thor as King of Asgard in the beginning of the first Thor movie, why h… https://t.co/qwRbUbNTfC'}, {'text': 'why is jane accidentally hitting thor with her car TWICE still one of the funniest things to happen in a marvel movie?'}, {'text': 'watching the Movie of the Bisexuals (thor ragnarok)'}, {'text': '@famouslyunknwn @twopoundsofraw I didn’t dislike any marvel movie except Thor 2. But doesn’t mean it’s not bottom t… https://t.co/RWkQK7oiaE'}, {'text': 'suriya  last movie kappan theatre release september 2019\n\n#Suriya40 relaese date september 2021 exactly 2 years 😲'}, {'text': '@ChanceTheSmashr You think Thor the Dark World was an actual movie???'}, {'text': 'Thirsty movie thoughts: Thor: Ragnarok\nI know Thor probably had a couch or something to sleep on, but what if he ha… https://t.co/7Yevj4HDr1'}, {'text': 'You know why people h8 capt. Marvel? Because it is the only superhero movie in the MCU with a female lead(as of now… https://t.co/MqCEtCZM35'}, {'text': 'omfgggg mcu stans tearing into captain marvel on tiktok its simply not a perfect movie bur where is this energy for… https://t.co/W4pwEGEsCp'}, {'text': "RT @StephieSparda: Sia's new movie has a lower critic score than Cats. https://t.co/CjTQMZzqBX"}, {'text': "@WorldOfTigra @GailSimone Ragnarok is a very enjoyable movie. It's just not much of a Thor movie. And I don't think… https://t.co/GMiKgFgn9z"}, {'text': '@mattgoldey @GailSimone That wasn’t really possible at the time. The first Avengers movie came out 3 months before… https://t.co/URJDoqiJJQ'}, {'text': "RT @rameshlaus: . @Suriya_offl 's #Suriya40 movie pooja happening today. https://t.co/i90jnBr9Vq"}, {'text': 'I had ONE dream about Chris Hemsworth and now I’m watching every Thor movie tonight 😔'}, {'text': 'People love to talk about how the audience reaction to Captain America getting Thor’s hammer was their favorite mov… https://t.co/I7ourJv3ju'}, {'text': '@AdamUnger_7 Infinity War - flew back from a bachelor party on a red eye &amp; went to a packed upper west side theater… https://t.co/8IF4ZiYGW0'}, {'text': 'Thor Ragnorak is a top 3 Marvel movie. Don’t @ me'}, {'text': 'I will never get over the fact that like 90% of the first Thor movie is a Dutch angle therefore rendering the use o… https://t.co/lemb20jUcn'}, {'text': "I feel like an all-time comedy movie hasn't hit the theaters since like, idk, Superbad? Just feels like they're so… https://t.co/9cdVcdMxT5"}]

Negative text sample: would have a hard time sitting through this one have a hard time sitting through this one aggressive
Positive text sample: this quiet , introspective and entertaining independent is worth seeking . quiet , introspective and
Some what negative text sample: a series of escapades demonstrating the adage that what is good for the goose is also good for the g
Some what positive text sample: good for the goose good amuses this quiet , introspective and entertaining independent quiet , intro
Neutral text sample: a series of escapades demonstrating the adage that what is good for the goose a series a series of e

Positive review count :  9206
Negative Review count :  7072
Neutral review count :  9206
Somewhat Positive review count :  9206
Somewhat Negative  review count :  9206

Review: A series of escapades demonstrating the adage that what is good for the goose is also good for the gander , some of which occasionally amuses but none of which amounts to much of a story .

Negative prediction: 6.912374606071673e-88
Positive prediction: 1.0969731496915699e-88
Neutral prediction: 7.294929017178259e-94
Some what Positive prediction: 3.000531338098658e-89
Some what Negative prediction: 1.1713475574052657e-87

First 50 predictions
[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 4, 4]

=====Rating=====
3.8048780487804876
```

## **Modules of Proposed System**

### **Tweet Collection**

Twitter data can be accessed through the public API provided by the Twitter. These APIs can be accessed only by authentication requests, which must be signed with valid login ID and password. Twitter provides authentication keys for extractions of the tweets. We have to follow some steps to create Authentication keys.

i. Create application on twitter.  
ii. Manage Application
iii. Change the permissions to read and write.  
iv. Retrieve Authentication keys.

Tweet Extracted from twitter having complete information like date of tweet, tweet ID, user ID, re Tweet count etc.
Twitter API was used to fetch all the tweets related to a particular movie and all the news and comments related to a particular movie.

### **Tweet Classification**

- Tweets were tokenized into different tokens separated by space and compare each token with our predefined set of positive and negative bag of words.
- After the comparison of tokens we find the total number of positive and negative tokens in the tweet.
- Count the total number of positive and negative tokens in the tweet and label them as p and n respectively.
- Calculate the value of ratio as total number of positive tokens to the total number of positive and negative token.
- Naïve bayes classifier was used for sentiment analysis of a tweet, which will classify a tweet into various categories .
- For every word in the text, we get the number of times that word occurred in the reviews for a given class, add 1 to smooth the value, and divide by the total number of words in the class (plus the class_count to also smooth the denominator).

### **Rating Calculation**

Every label carries a rating score with it which is used to calculate the overall score of a movie based on labels of its tweets.
