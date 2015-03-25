mvm <-function(angle_vec,xfit,c){
    # angle_vec, xfit in radians, c components
    unit_circle_vec <- cbind(sapply(angle_vec,cos), sapply(angle_vec,sin))
    xfit_circle <- cbind(sapply(xfit,cos), sapply(xfit,sin))

    vm <- tryCatch(movMF(unit_circle_vec, k = c), error = function(e) NULL)

    if (!is.null(vm)) {
        bic <- BIC(vm)
        yfit <- dmovMF(xfit_circle,vm$theta,vm$alpha)

        comp_fit=list()    
        for (i in 1:c) {
          comp_fit[[i]] <- dmovMF(xfit_circle,vm$theta[i,],vm$alpha[i])
        }
    
        mu <- atan2(vm$theta[,2], vm$theta[,1])
        kappa <- sqrt(rowSums(vm$theta^2))
        weights <- vm$alpha
    
        return(list(mu,kappa,weights,yfit,comp_fit,bic))
    } else {return(c(rep(list(0),5),1000))}
}
