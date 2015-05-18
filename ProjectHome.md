**tools update:** <br /> asymmetric prior Latent Dirichlet Allocation (LDA) by c++ ——https://code.google.com/p/as-lda/<br />adaboost codes——http://code.google.com/p/simple-adaboost/
<br />
any problem please contact benwang177@gmail.com <br />

---

**introduction:** <br />
it is simple implement of gbdt(actually gradient boost regression tree)--compared to others, parallelization with tbb lib.<br />

this project can be easily used in regression problem or point wise learning to rank problem(l2r), and can be good material to know about gbdt.<br />

there are several more features:<br />
1. support data sample configure<br />
2. using tbb lib for parallelization(http://threadingbuildingblocks.org/)<br />
3. empty features in l2r data will be skipped (to avoid unnecessary cost)<br />
4. support two input format(l2r format and cvs format)<br />


---

**Usage** <br />
> -i  input file<br />
> -f  [cvs, l2r]  input format , default 'l2r'<br />
> -c  gbrt model config file , default './gbrt.conf' <br />
> -t or -p  act type(t for train,p for predict): <br />
> -m  model file , default './gbrt.model' <br />
> -d  max dimention(only for L2R format) <br />

**Example** <br />
train example: ./gbrt -t -i train\_l2r.dat -d 415<br />
predict example: ./gbrt -p -i test\_l2r.dat -d 415 -m gbrt.model<br />

**Input Format**<br />
> l2r: "-1      qid:1234        1:1     3:1     8:1"<br />
> csv: "1,3,-1" last column is label<br />


---

**configure example**<br />

max\_tree\_leafes=20<br />
feature\_subspace\_size=40<br />
use\_opt\_splitpoint=true<br />
learn\_rate=0.01<br />
data\_sample\_ratio=0.4<br />