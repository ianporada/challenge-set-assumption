# Run original stanford corenlp

# build

cd ~/Research/code/CoreNLP/
ant ; cd classes ; jar -cf ../stanford-corenlp.jar edu ; cd ..

# start

cd ~/Research/code/stanford-corenlp-custom/
cp ~/Research/code/CoreNLP/stanford-corenlp.jar ./stanford-corenlp-4.5.7.jar

# cp ~/Research/code/anaphora_induction/dcoref/coref.properties ./
# cp ~/Research/code/reference-coreference-scorers/scorer.pl ./

rm inference.properties
rm conlloutput-* ; rm evaluation_*

PROP_FNAME="inference.properties"
DATA_DIR=/Users/ianporada/Research/data/pcr
# CONFIG=conll2012_indiscrim_english_v4
CONFIG=pronominal_winogrande_default

echo "annotators = pos, lemma, ner, parse" > $PROP_FNAME
echo "dcoref.score = false" >> $PROP_FNAME
echo "dcoref.postprocessing = true" >> $PROP_FNAME
echo "dcoref.maxdist = -1" >> $PROP_FNAME
echo "dcoref.use.big.gender.number = true" >> $PROP_FNAME
echo "dcoref.big.gender.number = edu/stanford/nlp/models/dcoref/gender.map.ser.gz" >> $PROP_FNAME
echo "dcoref.replicate.conll = false" >> $PROP_FNAME
# echo "dcoref.conll.scorer = ./scorer.pl" >> $PROP_FNAME
echo "dcoref.logFile = evaluation_log.txt" >> $PROP_FNAME
echo "dcoref.conll2011 = $DATA_DIR/$CONFIG/split=test_localcontext=True.v4_auto_conll" >> $PROP_FNAME

java -cp "*" edu.stanford.nlp.dcoref.SieveCoreferenceSystem -props "inference.properties"

##

OUT=~/Research/data/pcr_e2e/$CONFIG/inference/
mkdir -p $OUT
cp conlloutput-* $OUT
cp evaluation_* $OUT