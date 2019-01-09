getwd()
setwd("C:/Users/dell/testing")


#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\bank//////////////////////////////////////////////////////////////#

library(rgl)
library(FactoMineR)
library(arules)
library(arulesViz)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(rpart)
library(party)
library(tree)
library(ROCR)
library(kknn)
library(e1071)
library(ROCR)
library(nnet)
library(MASS)
library(caret)


#importation des données
bankf<-read.csv(file="bank.csv",header=T,sep=";",dec=".")

yes=bankf[which(bankf$y == 'yes'),]
dim(yes)
no=bankf[which(bankf$y == 'no'),]
dim(no)

summary(no)
summary(yes)

#choix des varibles 
bank=bankf[,c(1:8,17)]
head(bank)
str(bank)
dim(bank)




#*************************************************************ACP**************************************************************#

#on selectionne les variables quantitative
bank.active=bank[,c(1,6)]

pairs(bank)
#pour les variables qualitative leurs interpretation n'est pas trés indiqué prenons un exemple
#pour la variable material et blance; ici pour chaque statut le solde annuel (balance) varie 
#=>pour les variables qualitative en général, le nuage de point n’est pas vraiment un nuage mais une répartition de l’effectif en des droites
cor(bank.active)
#=>les varibles age et balance ne sont pas corollé donc la variation de l'un ne depend pas de l'autre.


######partie 3D
pc <- princomp(bank.active, cor=TRUE, scores=TRUE)
attributes(pc)
summary(pc)
biplot(pc)
plot3d(pc$scores[,1:2], col = as.numeric(bank$y))
text3d(pc$scores[,1:2],texts=rownames(bank))
text3d(pc$loadings[,1:2], texts=rownames(pc$loadings), col="blue")
coords <- NULL
for (i in 1:nrow(pc$loadings)) {
  coords <- rbind(coords, rbind(c(0,0,0),pc$loadings[i,1:2]))
}
lines3d(coords, col="blue", lwd=2)
######

#recuperation des valeur propre des données centrées et reduites
val.propres<- pc$sdev^2

#on represente graphiquement la variation des valeurs propres pour choisir k la plus optimale qui maximise la representation des données
plot(1:2,val.propres,type="b",ylab="Valeurs propres",xlab="Composante",main="Scree plot")


#ici le choix de k est evident on a que deux varible mais normlement on doit utiliser soit le critere de coude, soit 
#le critere de kaiser qui mentionne qu'on doit prendre les valeur propre qui sont superieur à 1
res.pca = PCA(bank, scale.unit=TRUE,quali.sup=c(2:5,7:9), ncp=2, graph=T)
attributes(res.pca)
res.pca$eig
res.pca$var
res.pca$ind

cor(res.pca$eig,)
x11()
plot(res.pca,cex=0.2,shadow=TRUE,choix="ind",axes=c(1,2))
plot.pca(res.pca)

#Interpretation:
#on a l'axe 1 represente 54% de l'information et on a les deux varibles age et balance(solde annuel) 
#sont corrolé positivement avec l'axe1 
#donc l'axe 1 represente les individus ayant un solde annuel et un age  plus élevé par rapport aux autres individus 

#**************************************************************Kmeans***********************************************************#

#fonction pour centrer et reduire les colonnes
cr <- function(a){return((a-mean(a))/sqrt(var(a)))}

bank.cr <- apply(bank.active,2,cr)

#choix de k par l'evalution de l'inertie
inertie.expl <- rep(0,times=10)
for (k in 2:10){
clus <- kmeans(bank.cr,centers=k,nstart=5)
inertie.expl[k] <- clus$betweenss/clus$totss
}

plot(1:10,inertie.expl,type="b",xlab="Nb. de groupes",ylab="% inertie expliquée")

#on choisira k=2 

#application de la methode kmeans
bank.kmeans <- kmeans(bank.cr,centers=2)
groupekm<-bank.kmeans$cluster

#####partie3D
bank$cluster <- as.factor(bank.kmeans$cluster)
plot3d(pc$scores[,1:2], col=bank$cluster, main="k-means clusters")
#MAJ élimination de la variable cluster 
bank=bankf[,c(1:8,17)]
#############


head(bank)
#table de contingence avec y:
tkmens=table(bank$y,groupekm) #linge*colonne

#interpretation:
#taux de clasification pour no : 45%%
#%individu mal classe: 55%

#taux de clasification pour yes : 51%
#%individu mal classe: 49%


#taux d'erreur
errkmeans <- (1 - sum(diag(tkmens))/sum(tkmens))*100
errkmeans 
#la classification avec kmeans nou donne un pourcentage d'individu mal classé de 52% c'est pour cela qu'on précedera
#a l'application de cah pour voir si on obtient un taux d'erreur inferieur a celui de la methode kmeans 
head(bank)
kmjob=table(bank$job,groupekm)
kmmar=table(bank$marital,groupekm)


#**************************************************************CAH***********************************************************#
#faire la matrice de distance
d <- dist(bank.cr,method = "euclidean") 

#on applique l'algorithme de clasification ascendante
#on choisit la methode de classification ward pour eviter l'effet de chaine(pour eviter d'avoir un dendograme qui ne permet pas de faire le bon partionement)
cah <- hclust(d,method="ward.D2") 

#on affiche le dendograme
plot(cah,hang=-1,cex=0.75) 

#choisir nombre de class direct sur le plot
rect.hclust(cah,k=2) 

# affichage des classes sur la commande
class <- cutree(cah,k=2) 

#table de contingence avec y:
tcah=table(bank$y,class) #linge*colonne

#interpretation:
#taux de clasification pour no : 47.72%
#%individu mal classe: 52.28%

#taux de clasification pour yes : 50.96%
#%individu mal classe: 49.03%

#taux d'erreur
errkmeans <- (1 - sum(diag(tcah))/sum(tcah))*100

#la classification avec kmeans nous donne un pourcentage d'individu mal classé de 50.44% donc le cah
#nous donne une meilleure classification 

#interpretation generale:
#les methodes de classification nous ont montré une propotionnalite entre les individus avec comme reponse yes et no
#donc chaque classe on trouve presque des valeurs propotionnelle donc on peut conclure que la varible blance(solde annuele)
#n'est pas la plus decisive pour savoir si un individu va renouveler son contrat ou pas
#ça sert à rien aussi de faire les tables de contigence pour les autres varibles (exp:housing,iona...) pour caracteriser
#chaque segment vu que la segmentation n'a pas donné un resultat optimal.  

#*********************************************************hcpc*****************************************************************#
res.pca2 = PCA(bank.active, scale.unit=TRUE, ncp=2, graph=T)

#application de la methode cah ur le resultat de l'acp
res.hcpc = HCPC(res.pca2)
#=>L'arbre hiérarchique suggère une partition en trois classes 

#*********************************************************Apriori*****************************************************************#



#recuperation des individus ayant comme reponse yes
yes=bankf[which(bankf$y == 'yes'),]
yes=yes[,c(5,7:8,17)]
head(yes)

#conversion du dataframe en transactions
tranyes<-as(yes,"transactions")
tranyes

#Exploration des données en utilisant l’alghorithme Apriori
aprioyes=apriori(tranyes, parameter = list( supp=0.001, conf=0.5))

summary(aprioyes)

#ordonner les règles par la valeur de l'indicateur lift
aprioyes=sort(aprioyes,by="lift")

#slectionner les 80 premiére régles
inspect(aprioyes[0:80])


plot(aprioyes,interactive=T)
#interprattion:
#on a pas obtenu une régle qui a comme resultat y= yes et comme indicateur de performance (lift)>1
#on a trouvé des régles avec comme resultat y=yes mais leur lift=1 qui signifie une correlation nulle


#************************************************classification arbre**********************************************************#



#entrêpot d'apprentisage
app=bank[1:800,]
#entrêpot de test
test=bank[800:1011,]
head(bank)

######rpart######
#on choisie les attribues qu'on doit utiliser pour la prediction
tree = rpart(y~age+default+balance+housing+loan, data=app, minsplit=50)
x11()
fancyRpartPlot(tree)

#interpretation:
#d'aprest le resultat obtenue on remarque que la varible la plus decisive est housing(credit immo)
#on a obtenu aussi 5 regle decisionelle on remarque que ya des regle avec un pourcentage tres faibe qui varie entre 3% et 9% donc 
#leur interpretation ne sera d'aucune utilite
#donc il reste deux regle a interprete
#1ere ceux qui non pas de credit immo(housing) et ceux qui non pas non plus un credit personnel sont supectibe de renuveler leur contra bancaire
#2em si housing(yes) et balance<1598 alors y = no

#application du model rpart sur lentrepot de test
predict1 <-predict(tree ,test ,type="class")
table(test$y,predict1)
#taux d'erreur du modele:48.11%
#################



#####party######
ctree<-ctree(y~age+default+balance+housing+loan,app )
plot(ctree, uniform = TRUE, branch = 0.5, margin = 0.1)
text(ctree, all = FALSE, use.n = TRUE)

#interpretation:
#d'aprest le resultat obtenue on remarque que la varible la plus decisive est housing(credit immo) "comme precedament avec rpart" 
#on a 3 regle decisive:
#1er : si housing(no) et ioan(no) on a 63% des individu avec y=yes et 37% des individu avec y=no 
#2eme: si housing(no) et ioan(yes) on a 38% des individu avec y=yes et 62% des individu avec y=no 
#3eme: si housing(yes) alors 41%  des individu avec y=yes et 59% des individu avec y=no 


#application du model ctree sur lentrepot de test
predict2<-predict(ctree,test,type="response")
table(test$y,predict2)
#taux d'erreur du modele:43.86%
#################


#####tree######
tree3<-tree(y~age+default+balance+housing+loan,app )
plot(tree3, uniform = TRUE, branch = 0.5, margin = 0.1)
text(tree3, all = FALSE, use.n = TRUE)

#interpretation:
#d'aprest le resultat obtenue on remarque que la varible la plus decisive est housing(credit immo) 
#on a 4 regle decisive:
#1er : si housing(no) et age>58.5 alors y=yes
#2eme : si housing(no) et age<58.5 alors y=no
#3eme : si housing(no) et ioan(yes) alors y=yes
#4eme : si housing(no) et ioan(no)  alors y=no


#application du model ctree sur lentrepot de test
predict3<-predict(tree3,test,type="class")
table(test$y,predict3)
#taux d'erreur du modele:44.81%
#################



pred1 <- prediction(as.numeric(predict1), test$y)
perf1 <- performance(pred1, "tpr","fpr")
pred2 <- prediction(as.numeric(predict2), test$y)
perf2 <- performance(pred2, "tpr","fpr")
pred3 <- prediction(as.numeric(predict3), test$y)
perf3 <- performance(pred3, "tpr","fpr")

plot(perf1,col="red")
par(new=T)
plot(perf2 ,col="blue")
par(new=T)
plot(perf3 ,col="green")
par(new=T)

#interpretation:
#d'apres le graphique obtenu le meilleur clasement corespond au modele 2 qui est celui de party



#*****************************************************classification kknn*******************************************************#


kknn <- kknn(y ~.,train=app,test=test,k =3, distance = 1)# on change le taux de k
summary(kknn)
attributes(kknn)

tableknn = table(test$y,kknn$fit)
#taux d'erreur
errkmeans <- (1 - sum(diag(tableknn ))/sum(tableknn ))*100
errkmeans # taux d'erreur :49.05%

#******************************************************classification SVM*******************************************************#


svm.basic <- svm(y ~., data=app)
predict.svm.basic <-  predict(svm.basic,test,type="class")
t1 = table(predict.svm.basic,test$y )

err1 <- 1 - sum(diag(t1))/sum(t1)
err1 # taux d'erreur :44.8%


plot(predict.svm.basic)
prop.table(t1)

svmModel <- svm(y~., data=app,kernel="linear")
predict.svm.linear <-  predict(svmModel,newdata= test)
t2 = table(predict.svm.linear,test$y )

err2 <- 1 - sum(diag(t2))/sum(t2)
err2 # taux d'erreur :45.7%

svm.sigmo <- svm(y~., data=app,kernel="sigmoid")
predict.svm.sigmo <-  predict(svm.sigmo,newdata= test)
t3 = table(predict.svm.sigmo,test$y)

err3 <- 1 - sum(diag(t3))/sum(t3)
err3 # taux d'erreur :44.3%



#pre
pred11 <- prediction(as.numeric(predict.svm.basic), test$y)
perf11 <- performance(pred11, "tpr","fpr")
pred22 <- prediction(as.numeric(predict.svm.linear), test$y)
perf22 <- performance(pred22, "tpr","fpr")
pred33 <- prediction(as.numeric(predict.svm.sigmo), test$y)
perf33 <- performance(pred33, "tpr","fpr")


plot(perf11,col="red")
par(new=T)
plot(perf22 ,col="blue")
par(new=T)
plot(perf33 ,col="green")
par(new=T)

#on choisira le model en vert qui corespond au model sigmoide de svm

le tauxe d'erreur pour la methode knn est de 49.05% et le taux d'erreur de la methode svm(model sigmoide) a un taux d'erreu de 44.3%
donc la methode svm est celle qui nous donne un meilleur resultat 


#*******************************************************reseau de neurone*******************************************************#



rn = nnet(y~.,data=bank, size=2)
pred = predict(rn , newdata= bank, type="class")
pred

table(bank$y, pred)
rn = nnet(y~.,data=bank, size=2)
summary(rn)

K<-150
res<-numeric(K)
for(k in 1:K){
model=nnet(y~.,data=bank, size=k, skip=FALSE)
pred = predict(model , newdata= bank, type="class")
mc<-table(bank$y, pred)
error<-1-sum(diag(mc))/sum(mc)
res[k]<-error
}

plot(res)
#interpret:
#le modele de reseau deneurone se stabilise apres k=47


#**********************************************************scoring**************************************************************#

app.lda=lda(y~.,data=app)
lda.pred <- predict(app.lda,app)
attributes(lda.pred)
on=table(app$y,lda.pred$class)
plot(on)

284/(665+287)
21/(21+27)

app$c1<- cbind(lda.pred$posterior[,1:1])
app$c2<- cbind(lda.pred$posterior[,2:2])

datao <- app[order(app$c1,decreasing=T),]

head(datao)


datao[1:11,] 


lifto <- lift(factor(y) ~ lda.pred$posterior[, 1], data = app)
lifto
attributes(lifto)

xyplot(lifto,value = c(60))

#interpretation:
#if faut envoyer un courrier aux 47% premiers individus de la base pour atteindre 60% des individu avec reponse no



#********************************************************************************************************************************#



#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\operateur//////////////////////////////////////////////////////////////#

#importation des données
data<-read.csv(file="churn.arff.csv",header=T,sep=",",dec=".")
data
summary(data)

yes=data[which(data$LEAVE == 'LEAVE'),]
dim(yes)
no=data[which(data$LEAVE == 'STAY'),]
dim(no)
 
data.quanti=data[,c(2:8)]
head(data.quanti)
str(data.quanti)
dim(data.quanti)

#*************************************************************ACP**************************************************************#
summary(data.active)
#on selectione les variables quantitative
data.active=data[,c(2:8)]

pairs(data)
#pour les varibles qualitative leur interpretation n'est pas tres indique prenant 
#pour les variables qualitative en generale, le nuage de point n’est pas vraiment un nuage mais une répartition de l’effectif en des droites
cor(data.active)
#=>les varibles handsetprie et income sont tres corole (0,72) donc la variation de l'un depend de l'autre. (plus income augmente plus handset price augmente)


######partie 3D
pc <- princomp(data.active, cor=TRUE, scores=TRUE)
attributes(pc)
summary(pc)
biplot(pc)
library(rgl)
plot3d(pc$scores[,1:2], col = as.numeric(data$LEAVE))
text3d(pc$scores[,1:2],texts=rownames(data))
text3d(pc$loadings[,1:2], texts=rownames(pc$loadings), col="blue")
coords <- NULL
for (i in 1:nrow(pc$loadings)) {
  coords <- rbind(coords, rbind(c(0,0,0),pc$loadings[i,1:2]))
}
lines3d(coords, col="blue", lwd=2)
######

#recuperation des valeur propre des donne centre et reduite
val.propres<- pc$sdev^2

#on represente graphiquement la variation des valeurs propres pour choisr k la plus optimale qui maximise la representation des donnes
plot(1:7,val.propres,type="b",ylab="Valeurs propres",xlab="Composante",main="Scree plot")


library(FactoMineR)
#pour le choix de k on doit utiliser soit le critere de coude, soit 
#le critere de kaiser qui mentionne qu'on doit prendre les valeur propre qui sont superieur a 1
res.pca = PCA(data, scale.unit=TRUE,quali.sup=c(1,9:12), ncp=3,graph=T)
attributes(res.pca)
res.pca$eig
res.pca$var
res.pca$ind

cor(res.pca$eig,)
x11()
plot(res.pca,cex=0.2,shadow=TRUE,choix="ind",axes=c(1,2))



#Interpretation:
#on a l'axe 2 represente 24% de l'information et on a les deux varibles income et handsetprice(solde anuelle) 
#sont corole positivement 
#donc l'axe 2 represente les individu ayant un salaire et une valeur de maison plus eleve.  

#**************************************************************Kmeans***********************************************************#

#fonction pour centrer et reduire les colonne
cr <- function(a){return((a-mean(a))/sqrt(var(a)))}

data.cr <- apply(data.active,2,cr)

#application de la methode kmeans
data.kmeans <- kmeans(data.cr,centers=2)
groupekm<-data.kmeans$cluster

#1ere methode: choix de k par l'evalution de linertie
inertie.expl <- rep(0,times=10)
for (k in 2:10){
clus <- kmeans(data.cr,centers=k,nstart=5)
inertie.expl[k] <- clus$betweenss/clus$totss
}

plot(1:10,inertie.expl,type="b",xlab="Nb. de groupes",ylab="% inertie expliquée")

#2ere methode: indice de calinski 
library(fpc)
sol.kmeans <- kmeansruns(data.cr ,krange=2:10,criterion="ch")
plot(1:10,sol.kmeans$crit,type="b",xlab="Nb. de groupes",ylab="Silhouette")

#=> on utilison les deux criteres on obtient la valeur de k egale a 3

#####partie3D
data$cluster <- as.factor(data.kmeans$cluster)
plot3d(pc$scores[,1:2], col=bank$cluster, main="k-means clusters")


#table de contingence avec y:
table(data$LEAVE,groupekm) #linge*colonne

#interpretation:
#taux de clasification pour no : 53%%
#%individu mal classe: 47%

#taux de clasification pour yes : 51%
#%individu mal classe: 49%

#
# l'application de cah pour voir si on obtient un taux d'erreur inferieur a celui de la methode kmeans 

#**************************************************************CAH***********************************************************#
#faire la matrice de distance
d <- dist(data.cr,method = "euclidean") 

#on applique l'algrorithle de clasification ascandante
#on choisi la methode de classification ward pour eviter l'effet de chaine(pour eviter d'avoir un dendograme qui ne permet pas de fait le bon partionement)
cah <- hclust(d,method="ward.D2") 

#on affiche le d'endograme
plot(cah,hang=-1,cex=0.75) 

# choisir nombre de class direct sur le plot
rect.hclust(cah,k=2) 

# affichage des classe sur la comande
class <- cutree(cah,k=2) 

#table de contingence avec LEAVE:
table(data$LEAVE,class) #linge*colonne

#interpretation:
#taux de clasification pour LEAVE : 47.72%
#%individu mal classe: 52.28%

#taux de clasification pour STAY : 50.96%
#%individu mal classe: 49.03%

#la classification avec kmeans nou donne un pourcentage d'individu mal classe de 50.65% donc le cah
#nous donne une meilleur clasification 


#*********************************************************hcpc*****************************************************************#
res.pca2 = PCA(data.active, scale.unit=TRUE, ncp=2, graph=T)

res.hcpc = HCPC(res.pca2)
#=>L'arbre hiérarchique suggère une partition en trois classes 




#************************************************classification arbre**********************************************************#

library(rattle)
library(rpart.plot)
library(RColorBrewer)

#entrêpot d'apprentisage
app=data[1:15000,]
#entrêpot de test
test=data[15001:20000,]

######rpart######
library(rpart)
#on choisie les attribues qu'on doit utilise pour la prediction
tree = rpart(LEAVE~., data=app)
x11()
fancyRpartPlot(tree)

#interpretation:
#d'aprest le resultat obtenue on remarque que la varible la plus decisive est house(Value of dwelling)
#on a obtenu aussi 6 regle decisionelle
#les deux regle a interpreter avec un pourcentage un peu eleve par rapport au autres regles:
#les clients qui ont une maison de valeur importante (> a 600e3) avec un salaire pas tres important(> a 100e3) sont des clients fideles
#2em si valeur de house pas tres importante et moyenne des appeles en minutes par mois superieur a 98, client scuseptible de churner(changer d'operateur)

#application du model rpart sur lentrepot de test
predict1 <-predict(tree ,test ,type="class")
table(test$LEAVE,predict1)
#taux d'erreur du modele:29.96%
#################



#####party######
library(party)
ctree<-ctree(LEAVE~.,app )
plot(ctree, uniform = TRUE)
text(ctree, all = FALSE, use.n = TRUE)
 


#application du model ctree sur lentrepot de test
predict2<-predict(ctree,test,type="response")
table(test$LEAVE,predict2)
#taux d'erreur du modele:29.08%
#################


#####tree######
library(tree)
tree3<-tree(LEAVE~.,app )
plot(tree3, uniform = TRUE, branch = 0.5, margin = 0.1)
text(tree3, all = FALSE, use.n = TRUE)

#interpretation:
#d'aprest le resultat obtenue on remarque que la varible la plus decisive est house commes les autres modele 
#regle decisive:
#1er : si housing>600480 et salaire>100012 alors churn(LEAVE)
#2eme : si housing<600480 et valeur overcharges<98.5 alors churn(LEAVE)
#3eme : si housing<600480 et valeur overcharges<98.5 alors churn(LEAVE)



#application du model ctree sur lentrepot de test
predict3<-predict(tree3,test,type="class")
table(test$LEAVE,predict3)
#taux d'erreur du modele:29.96%
#################

library(ROCR)

pred1 <- prediction(as.numeric(predict1), test$LEAVE)
perf1 <- performance(pred1, "tpr","fpr")
pred2 <- prediction(as.numeric(predict2), test$LEAVE)
perf2 <- performance(pred2, "tpr","fpr")
pred3 <- prediction(as.numeric(predict3), test$LEAVE)
perf3 <- performance(pred3, "tpr","fpr")

plot(perf1,col="red")
par(new=T)
plot(perf2 ,col="blue")
par(new=T)
plot(perf3 ,col="green")
par(new=T)

#interpretation:
#d'apres le graphique obtenu le meilleur clasement corespond au modele 2 qui est celui de party



#*****************************************************classification kknn*******************************************************#


library(kknn)


kknn <- kknn(LEAVE ~.,train=app,test=test,k =3, distance = 1)# on change le taux de k
summary(kknn)
attributes(kknn)

tableknn = table(test$LEAVE,kknn$fit)




#******************************************************classification SVM*******************************************************#
library(e1071)

svm.basic <- svm(LEAVE ~., data=app)
summary(svm.basic)
attributes(svm.basic)

predict.svm.basic <-  predict(svm.basic,test,type="class")
t1 = table(predict.svm.basic,test$LEAVE )

plot(predict.svm.basic)
prop.table(t1)

#*******************************************************reseau de neurone*******************************************************#


library(nnet)

rn = nnet(LEAVE~.,data=data, size=2)
pred = predict(rn , newdata= data, type="class")
pred

table(data$LEAVE, pred)
rn = nnet(LEAVE~.,data=data, size=2)
summary(rn)

K<-100
res<-numeric(K)
for(k in 1:K){
model=nnet(LEAVE~.,data=data, size=k, skip=FALSE)
pred = predict(model , newdata= data, type="class")
mc<-table(data$LEAVE, pred)
error<-1-sum(diag(mc))/sum(mc)
res[k]<-error
}

plot(res)

#le modele de stabilise a partir de k=50
#**********************************************************scoring**************************************************************#
library(MASS)
app.lda=lda(LEAVE~.,data=app)
lda.pred <- predict(app.lda,app)
attributes(lda.pred)
on=table(app$LEAVE,lda.pred$class)
plot(on)

284/(665+287)
21/(21+27)

app$c1<- cbind(lda.pred$posterior[,1:1])
app$c2<- cbind(lda.pred$posterior[,2:2])

datao <- app[order(app$c1,decreasing=T),]

head(datao)
library(caret)

datao[1:11,] 
library(caret)

lifto <- lift(factor(LEAVE) ~ lda.pred$posterior[, 1], data = app)
lifto
attributes(lifto)

xyplot(lifto,value = c(75))


#selon la courbe lift, par le ciblage de 60% des clients on peut atteindre 75% de Customer churn
prediction



#********************************************************************************************************************************#










































