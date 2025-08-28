#!/bin/bash

#	https://icon.colorado.edu/#!/networks
#	https://string-db.org/


echo -e '\nCreate work directories\n'

CURR_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )

echo $PWD
mkdir $PWD/cache
mkdir $PWD/data
cp $CURR_DIR/data/*.zip $PWD/data/
cd $PWD/data/

#==========================================================
echo -e '\nSetup of "Synthetic Datasets"\n'

wget https://zenodo.org/records/7749500/files/synthetic.zip
unzip -oq synthetic
unzip -oq noise

for OLD_FOLDER in synthetic/*; do
	NEW_FOLDER="${OLD_FOLDER/'synthetic_'/''}"
	mv $OLD_FOLDER $NEW_FOLDER
	
	for FILE in $NEW_FOLDER/graphs/*.txt; do
		LABEL=$(echo $FILE | grep -oP '[A-z]+[0-9]*(?=_)')
		
		if [[ -z "$LABEL" ]]; then
			break
		fi
		
		mkdir -p $NEW_FOLDER/$LABEL
		mv $FILE "${FILE/'graphs'/$LABEL}"
	done
	
	rm -rf $NEW_FOLDER/graphs
	rm -rf $NEW_FOLDER/labels
done

for OLD_FOLDER in noise/*; do
	NEW_FOLDER=${OLD_FOLDER/'noise/ruido'/'synthetic/noise'}
	mv $OLD_FOLDER $NEW_FOLDER
	
	for OLD_FILE in $NEW_FOLDER/graphs/*.txt; do
		LABEL=$(echo $OLD_FILE | grep -oP '[A-z]+[0-9]*(?=_.*sf.*)')
		NEW_FILE=${OLD_FILE/'graphs'/$LABEL}
		NEW_FILE=${NEW_FILE/'sf_'/'n=1000_i='}
		
		if [[ -z "$LABEL" ]]; then
			break
		fi
		
		mkdir -p $NEW_FOLDER/$LABEL
		mv $OLD_FILE $NEW_FILE
		sed -i '1,3d' $NEW_FILE
	done
	
	rm -rf $NEW_FOLDER/graphs
	rm -rf $NEW_FOLDER/labels
done

rm 'noise' -r
rm 'synthetic/noise' -r
rm 'synthetic/noise=0' -r

for FOLDER in synthetic/*/*; do
	OLD_LABEL=${FOLDER##*/}
	NEW_LABEL=${OLD_LABEL/'barabasi'/'BA'}
	NEW_LABEL=${NEW_LABEL/'watts'/'WS'}
	NEW_LABEL=${NEW_LABEL/'erdos'/'ER'}
	NEW_LABEL=${NEW_LABEL/'geo'/'GEO'}
	NEW_LABEL=${NEW_LABEL/'mendes'/'DM'}
	NEW_LABEL=${NEW_LABEL/'MEN'/'DM'}
	NEW_LABEL=${NEW_LABEL/'BANL5'/'BANL05'}
	
	if [ "$NEW_LABEL" != "BANL20" ]; then
		NEW_LABEL=${NEW_LABEL/'BANL2'/'BANL20'}
		NEW_LABEL=${NEW_LABEL/'BAnl2'/'BANL20'}
	fi
	
	if [ "$OLD_LABEL" = "$NEW_LABEL" ]; then
		continue
	fi
	
	mkdir -p ${FOLDER/$OLD_LABEL/$NEW_LABEL}
	
	for OLD_FILE in $FOLDER/*; do
		NEW_FILE=${OLD_FILE//$OLD_LABEL/$NEW_LABEL}
		if [ "$OLD_FILE" != "$NEW_FILE" ]; then
			mv $OLD_FILE $NEW_FILE
		fi
	done
	
	rmdir $FOLDER
done

mv synthetic/model synthetic/classic
mv synthetic aics-synthetic
rm *.zip



#==========================================================
echo -e '\nSetup of "KEGG : metabolic networks"\n'

wget https://zenodo.org/record/7749514/files/real.zip
unzip -oq real
mv real kegg-metabolic
rm *.zip



#==========================================================
echo -e '\nSetup of "Ego : social circles"\n'

SOURCE_URL="https://snap.stanford.edu/data"
TARGET_DIR="snap-ego/social-circles"

wget -P $TARGET_DIR $SOURCE_URL/readme-Ego.txt

for FILE in "gplus" "twitter" "facebook"; do
	echo -e " - $FILE.tar.gz"
	wget -P $TARGET_DIR $SOURCE_URL/$FILE.tar.gz
	tar -xzf $TARGET_DIR/$FILE.tar.gz --directory=$TARGET_DIR
	rm $TARGET_DIR/$FILE.tar.gz
done


#==========================================================
echo -e '\nSetup of some datasets from "TUDataset"\n'

#	https://chrsmrrs.github.io/datasets/
SOURCE_URL="https://www.chrsmrrs.com/graphkerneldatasets"

TARGET_DIR="tud-social"

for DATASET in "COLLAB" "IMDB-BINARY" "IMDB-MULTI" "REDDIT-BINARY" "REDDIT-MULTI-5K" "REDDIT-MULTI-12K" "deezer_ego_nets" "github_stargazers" "twitch_egos"; do
	wget -P "$TARGET_DIR" "$SOURCE_URL/$DATASET.zip"
	echo -e " - $DATASET.zip"
	unzip -oq $TARGET_DIR/$DATASET -d $TARGET_DIR
done

TARGET_DIR="tud-bioinformatics"

for DATASET in "DD" "ENZYMES" "PROTEINS" "MUTAG" "NCI1" "NCI109"; do
	wget -P "$TARGET_DIR" "$SOURCE_URL/$DATASET.zip"
	echo -e " - $DATASET.zip"
	unzip -oq $TARGET_DIR/$DATASET -d $TARGET_DIR
done

rm tud*/*/*.txt~
rm tud*/*.zip


#==================================================
#echo -e '\n>> setup of "STRING : protein-protein interactions (partial)"\n'

#wget https://stringdb-downloads.org/download/protein.links.v12.0.txt.gz		# dataset completo
#wget -P data/string/ https://stringdb-downloads.org/download/species.v12.0.txt

#for domain in "Archaea" "Bacteria"; do
#	echo $domain
#	regexp="^[0-9]+(?=\t.+\t$domain$)"
#	grep -oP $regexp data/string/species.v12.0.txt
#	break
#done

# homo sapiens
#wget -P data/string-protein/ https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz
#for f in data/string/*.gz; do gzip -d "$f"; done


#==================================================
for COLLECTION in *; do
	for OLD_FOLDER in $COLLECTION/*; do
		NEW_FOLDER=$(echo $OLD_FOLDER | sed -r 's/[_]+/-/g')
		if [ "$OLD_FOLDER" != "$NEW_FOLDER" ]; then
			echo "renaming: $OLD_FOLDER > $NEW_FOLDER"
			mv $OLD_FOLDER $NEW_FOLDER
		fi
		for OLD_CONTENT in $NEW_FOLDER/*; do
			NEW_CONTENT=$(echo $OLD_CONTENT | sed -r "s|${OLD_FOLDER##*/}|${NEW_FOLDER##*/}|g")
			if [ "$OLD_CONTENT" != "$NEW_CONTENT" ]; then
				echo "renaming: $OLD_CONTENT > $NEW_CONTENT"
				mv $OLD_CONTENT $NEW_CONTENT
			fi
		done
	done
done
