# Ross-thick Li-sparse model in:
# Lucht, W., Schaaf, C. B., & Strahler, A. H. (2000). 
# An algorithm for the retrieval of albedo from space using semiempirical BRDF models. 
# IEEE Transactions on Geoscience and Remote Sensing, 38(2), 977-998.


# source("model.r")

# Hankui wrote it on Nov 21 2017 

# cos1 <- (1:10)/10
# cos2 <- cos1
# sin1 <- cos1
# sin2 <- cos1
# cos3 <- cos1
GetPhaang <- function(cos1, cos2, sin1, sin2, cos3)
{
	cosres = cos1 * cos2 + sin1 * sin2 * cos3;
	res    = acos(pmax(-1., pmin(1., cosres)));
	sinres = sin(res);

	list("cosres"=cosres, "res"=res, "sinres"=sinres)
}
# tan1 <- cos1
# tan2 <- cos1
# cos3 <- cos1

GetDistance <- function(tan1, tan2, cos3)
{
	temp = tan1 * tan1 + tan2 * tan2 - 2. * tan1 * tan2 * cos3;
	res  = sqrt(pmax(0., temp));

	res
}

GetpAngles <- function(brratio, tan1)
{
	tanp = brratio * tan1;
	# if (tanp < 0) tanp = 0.;
	tanp[ tanp < 0 ] = 0;
	angp = atan(tanp);
	
	sinp = sin(angp);
	cosp = cos(angp);
	list("sinp"=sinp, "cosp"=cosp, "tanp"=tanp)
}



GetOverlap <- function(hbratio, distance, cos1, cos2, tan1, tan2, sin3, overlap, temp)
{
	temp = 1. / cos1 + 1. / cos2;
	cost = hbratio * sqrt(distance * distance + tan1 * tan1 * tan2 * tan2 * sin3 * sin3) / temp;
	cost = pmax(-1., pmin(1., cost));
	tvar = acos(cost);
	sint = sin(tvar);
	overlap = 1. / PI * (tvar - sint * cost) * (temp);
	overlap = pmax(0., overlap);

	list("overlap"=overlap, "temp"=temp)
	
}

LiKernel <- function(hbratio, brratio, tantv, tanti, sinphi, cosphi, SparseFlag, RecipFlag)
	# double hbratio, brratio, tantv, tanti, sinphi, cosphi;
	# int SparseFlag, RecipFlag;
{
	# double sintvp, costvp, tantvp, sintip, costip, GetpAnglesi$tanp;
	# double phaangp, cosphaangp, sinphaangp, distancep, overlap, temp;

	GetpAnglesv <- GetpAngles(brratio, tantv); # list("sinp"=sinp, "cosp"=cosp, "tanp"=tanp)
	GetpAnglesi <- GetpAngles(brratio, tanti);
	GetPhaang <- GetPhaang(GetpAnglesv$cosp, GetpAnglesi$cosp, GetpAnglesv$sinp, GetpAnglesi$sinp, cosphi); #list("cosres"=cosres, "res"=res, "sinres"=sinres)
	distancep <- GetDistance(GetpAnglesv$tanp, GetpAnglesi$tanp, cosphi);
	GetOverlap <- GetOverlap(hbratio, distancep, GetpAnglesv$cosp, GetpAnglesi$cosp, GetpAnglesv$tanp, GetpAnglesi$tanp, sinphi);

	if (SparseFlag) {
		if (RecipFlag) {
			result = (GetOverlap$overlap - GetOverlap$temp) + 1. / 2. * (1. + GetPhaang$cosres) / GetpAnglesv$cosp / GetpAnglesi$cosp;
		} else {
			result = GetOverlap$overlap - GetOverlap$temp + 1. / 2. * (1. + GetPhaang$cosres) / GetpAnglesv$cosp;
		}
	} else {
		if (RecipFlag) {
			result = (1 + GetPhaang$cosres) / (GetpAnglesv$cosp * GetpAnglesi$cosp * (GetOverlap$temp - GetOverlap$overlap)) - 2.;
		} else {
			result = (1 + GetPhaang$cosres) / (GetpAnglesv$cosp * (GetOverlap$temp - GetOverlap$overlap)) - 2.;
		}
	}
	
	result
}

CalculateKernels <- function(tv, ti, phi)	
	# double *resultsArray;
	# double tv, ti, phi;
{
	# int /*currker,*/ SparseFlag, RecipFlag;
	# double cosphi, sinphi;
	# double costv, costi, sintv, sinti, tantv, tanti;
	# double phaang, sinphaang, cosphaang;
	# double rosselement;// , distance;
	# //int iphi;

	# resultsArray <- c(0,0,0);
	resultsArray <- matrix(, nrow=length(tv), ncol=3);

	resultsArray[, 1] = 1.;

	cosphi = cos(phi);

	costv = cos(tv);
	costi = cos(ti);
	sintv = sin(tv);
	sinti = sin(ti);
	GetPhaang <- GetPhaang(costv, costi, sintv, sinti, cosphi); #list("cosres"=cosres, "res"=res, "sinres"=sinres)
	rosselement = (PI / 2. - GetPhaang$res) * GetPhaang$cosres + GetPhaang$sinres;
	resultsArray[, 2] = rosselement / (costi + costv) - PI / 4.;

# /*finish rossthick kernal */

	sinphi = sin(phi);
	tantv = tan(tv);
	tanti = tan(ti);

	# // SparseFlag = TRUE;
	# // RecipFlag = TRUE;
	SparseFlag = 1;
	RecipFlag = 1;
	resultsArray[, 3] <- LiKernel(hbratio, brratio, tantv, tanti, sinphi, cosphi, SparseFlag, RecipFlag);

	resultsArray
}

calc_refl_noround <- function(pars, vzn, szn, raa)
{
	# MAX_PAR <- (10000) ; 
	# PAR_FILL_VALUE <- (32767)

	# if(pars[1] > MAX_PAR || pars[2] > MAX_PAR || pars[3] > MAX_PAR) {
		# nbar <- PAR_FILL_VALUE;
	# } else {
		# double nbarkerval[3];
		# double nbar;
		# //int k;

	nbarkerval <- CalculateKernels(vzn*DE2RA, szn*DE2RA, raa*DE2RA);

	# ref = nbarkerval[1] * pars[1] + nbarkerval[2] * pars[2] + nbarkerval[3] * pars[3];
	ref = nbarkerval %*%  pars;
	# if(nbar < 0 && nbar > -30) {
		# nbar = 0.0;
	# } else if (nbar < 0 || nbar > MAX_PAR)
		# nbar = PAR_FILL_VALUE;
	# } 

	ref

}

# Coeffients in  Roy, D. P., Zhang, H. K., Ju, J., Gomez-Dans, J. L., Lewis, P. E., Schaaf, C. B., Sun Q., Li J., Huang H., & Kovalskyy, V. (2016). 
# A general method to normalize Landsat reflectance data to nadir BRDF adjusted reflectance. 
# Remote Sensing of Environment, 176, 255-271.

# If you want a fixed solar zenith output for a given band, 
#subsitute the three parameters with those from MODIS BRDF products at the same location and same day

#Global 12 months
#pars_array <- rbind(
#			  c( 774, 372, 79), 
#			  c(1306, 580,178),
#			  c(1690, 574,227),
#			  c(3093,1535,330),
#			  c(3430,1154,453),
#			  c(2658, 639,387)
#			  )

# Coefficients for RED and NIR bands are substituted with CONUS 12 months
pars_array <- rbind(
  c( 774, 372, 79), 
  c(1306, 580,178),
  c(1131, 462,247),
  c(2869,1833,367),
  c(3430,1154,453),
  c(2658, 639,387)
)

brratio <- 1.0;
hbratio <- 2.0;
DE2RA <- 0.0174532925199432956; 
PI <- (3.1415926);




##******************************************************************************************************************
## the following is for testing nbar 
# sz <- 0; 
# bandindex <- 3; 
# filename <- paste("band",bandindex, "sz", sz, "output.txt", sep=""); 
# if (file.exists(filename)) file.remove(filename);
# write("sz	vz	raa	ref.vz(*10000)	nbar(*10000)	c-factor",file=filename,append=TRUE) ; 
# for (vz in c(-78:78)/10 ) {
	# raa <- 0;
	# if (vz <0) raa <- 180;
	# ref1 <- calc_refl_noround(pars_array[bandindex, ], abs(vz), sz, raa);
	# ref0 <- calc_refl_noround(pars_array[bandindex, ],       0, sz, raa);
	# texts <- paste(sprintf("%.0f",sz  ), "\t", sprintf("%.2f",abs(vz)), "\t", sprintf("%.0f",raa      ), "\t", 
		# sprintf("%.6f",ref1), "\t", sprintf("%.6f",   ref0), "\t", sprintf("%.6f",ref0/ref1), sep="")
	# cat(texts, "\n");
	# write(texts,file=filename,append=TRUE) ; 
# }



##******************************************************************************************************************
## the following is for testing nbar with Array
# sz <- 0; 
# bandindex <- 3; 
# filename <- paste("band",bandindex, "sz", sz, "output.txt", sep=""); 
# vza <- c(-78:78)/10;
# sza <- rep(sz, length(vza));
# raa <- vza;
# raa [vza>=0] <-   0;
# raa [vza< 0] <- 180;
# ref1 <- calc_refl_noround(pars_array[bandindex, ], abs(vza)  , sza, raa);
# ref0 <- calc_refl_noround(pars_array[bandindex, ], abs(vza)*0, sza, raa);
 
# if (file.exists(filename)) file.remove(filename);
# write("sz	vz	raa	ref.vz(*10000)	nbar(*10000)	c-factor",file=filename,append=TRUE) ;
 
# for (i in 1:length(vza) ) {
	# texts <- paste(sprintf("%.0f",sza[i]), "\t", sprintf("%.2f",abs(vza[i])), "\t", sprintf("%.0f",raa[i]      ), "\t", 
		# sprintf("%.6f",ref1[i]), "\t", sprintf("%.6f",   ref0[i]), "\t", sprintf("%.6f",ref0[i]/ref1[i]), sep="")
	# cat(texts, "\n");
	# write(texts,file=filename,append=TRUE) ; 
	# break;
# }









# close(fileConn)
# fileConn<-file(filename); 
	# writeLines(, fileConn)
	# cat(texts, file=fileConn, append=TRUE);

# GetpAngles <- GetpAngles(brratio, 6)