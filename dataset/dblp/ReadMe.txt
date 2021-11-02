DBLP 

source:
<https://data.gesis.org/sharing/#>!Detail/10.7802/1520
DBLP version: June 2017 <https://dblp.org/xml/release/> ([dblp-2017-06-01.xml.gz](https://dblp.org/xml/release/dblp-2017-06-01.xml.gz))
Aminer: v10 <https://www.aminer.org/citation>

Include four areas of data derived from 20 conferences:
	Database->1
	Data Mining->2
	Machine Learning->3
	Information Retrieval->4
No label->0

N_author=28702
n_label=4236
n_gender=15555

After extracting the largest connected component, and eliminating all-zero and very sparse features:
N_author=20111
n_label=3196
n_gender=11080
n_feature_dim=2491

gender information is extracted from dataset provided by MOHSEN JADIDI

dblp_csr_emb.npy: author embedding processed from author-term relations.
we extract 2530 out of 13214 terms with total frequency>=20 and appear in >=10 authors
dim=(28702,2530)


for lable binarization, we combine (2,3,4) as 0, and 1 as 1 (personal understanding that "database" may be more different from the rest three)