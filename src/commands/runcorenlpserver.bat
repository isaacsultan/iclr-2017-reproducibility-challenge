cd ../data/stanford-corenlp

java -Xmx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 100000 -encoding utf-8

pause

