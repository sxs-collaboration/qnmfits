import numpy as np
import qnm as qnm_loader


class qnm:
    """
    Class for loading quasinormal mode (QNM) frequencies and spherical-
    spheroidal mixing coefficients. This makes use of the qnm package,
    https://arxiv.org/abs/1908.10377
    """
        
    def omega(self, l, m, n, sign, chif, Mf=1):
        """
        Return a complex frequency, :math:`\omega_{\ell m n}(M_f, \chi_f)`,
        for a particular mass, spin, and mode. One or both of chif and Mf can
        be array_like, in which case an ndarray of complex frequencies is
        returned.
        
        Parameters
        ----------
        l : int
            The angular number of the mode.
            
        m : int
            The azimuthal number of the mode.
            
        n : int
            The overtone number of the mode.
            
        sign : int
            An integer with value +1 or -1, to indicate the sign of the real
            part of the frequency. This way any regular (+1) or mirror (-1)
            mode can be requested. Alternatively, this can be thought of as
            prograde (sign = sgn(m)) or retrograde (sign = -sgn(m)) modes.
            
        chif : float
            The dimensionless spin magnitude of the black hole.
            
        Mf : float or array_like, optional
            The mass of the final black hole. This is the factor which the QNM
            frequencies are divided through by, and so determines the units of 
            the returned quantity. 
            
            If Mf is in units of seconds, then the returned frequency has units 
            :math:`\mathrm{s}^{-1}`. 
            
            When working with SXS simulations and GW surrogates, we work in 
            units scaled by the total mass of the binary system, M. In this 
            case, providing the dimensionless Mf value (the final mass scaled 
            by the total binary mass) will ensure the QNM frequencies are in 
            the correct units (scaled by the total binary mass). This is 
            because the frequencies loaded from the qnm package are scaled by 
            the remnant black hole mass (Mf*omega). So, by dividing by the 
            remnant black hole mass scaled by the total binary mass (Mf/M), we 
            are left with Mf*omega/(Mf/M) = M*omega.
            
            The default is 1, in which case the frequencies are returned in
            units of the remnant black hole mass.
            
        Returns
        -------
        complex or ndarray
            The complex QNM frequency.
        """
        # Load the correct qnm based on the type we want
        m *= sign
        
        # Use the qnm package to get the qnm frequency
        qnm_func = qnm_loader.modes_cache(-2, l, m, n)
        omega, A, mu = qnm_func(chif)
            
        # Use symmetry properties to get the mirror mode, if requested
        if sign == -1:
            omega = -np.conjugate(omega)
        
        return omega/Mf
    
    def omega_list(self, modes, chif, Mf=1):
        """
        Return a frequency list, containing frequencies corresponding to each
        mode in the modes list (for a given mass and spin).
        
        Parameters
        ----------            
        modes : array_like
            A sequence of (l,m,n,sign) tuples to specify which QNMs to load 
            frequencies for. For nonlinear modes, the tuple has the form 
            (l1,m1,n1,sign1,l2,m2,n2,sign2,...).
            
        chif : float
            The dimensionless spin magnitude of the final black hole.
            
        Mf : float or array_like, optional
            The mass of the final black hole. See the qnm.omega docstring for
            details on units. The default is 1.
            
        Returns
        -------
        list
            The list of complex QNM frequencies.
        """
        # For each mode, call the qnm function and append the result to the
        # list
        
        # Code for linear QNMs:
        # return [self.omega(l, m, n, sign, chif, Mf) for l, m, n, sign in modes]
        
        # Code for nonlinear QNMs:
        return [
            sum([self.omega(l, m, n, sign, chif, Mf) 
                 for l, m, n, sign in [mode[i:i+4] for i in range(0, len(mode), 4)]
                 ]) 
            for mode in modes
            ]
    
        # Writen out, the above is doing the following:
            
        # return_list = []
        # for mode in modes:
        #     sum_list = []
        #     for i in range(0, len(mode), 4):
        #         l, m, n, sign = mode[i:i+4]
        #         sum_list.append(self.omega(l, m, n, sign, chif, Mf))
        #     return_list.append(sum(sum_list))
        # return return_list
        
    def mu(self, l, m, lp, mp, nprime, sign, chif):
        """
        Return a spherical-spheroidal mixing coefficient, 
        :math:`\mu_{\ell m \ell' m' n'}(\chi_f)`, for a particular spin and 
        mode combination. The indices (l,m) refer to the spherical harmonic. 
        The indices (l',m',n') refer to the spheroidal harmonic. The spin chif
        can be a float or array_like.
        
        Parameters
        ----------
        l : int
            The angular number of the spherical-harmonic mode.
            
        m : int
            The azimuthal number of the spherical-harmonic mode.
            
        lp : int
            The angular number of the spheroidal-harmonic mode.
            
        mp : int
            The azimuthal number of the spheroidal-harmonic mode.
            
        nprime : int
            The overtone number of the spheroidal-harmonic mode.
            
        sign : int
            An integer with value +1 or -1, to indicate the sign of the real
            part of the QNM frequency. If the mixing coefficient associated
            with a -1 QNM (i.e. a mirror mode) is requested, then symmetry 
            properties are used for the calculation.
            
        chif : float or array_like
            The dimensionless spin magnitude of the final black hole.
            
        Returns
        -------
        complex or ndarray
            The spherical-spheroidal mixing coefficient.
        """
        # There is no overlap between different values of m
        if mp != m:
            return 0
        
        # Load the correct qnm based on the type we want
        m *= sign
        mp *= sign
        
        # Our functions return all mixing coefficients with the given 
        # (l',m',n'), so we need to index it to get the requested l
        if abs(m) > 2:
            index = l - abs(m)
        else:
            index = l - 2
            
        # Use the qnm package to get the qnm frequency
        qnm_func = qnm_loader.modes_cache(-2, lp, mp, nprime)
        omega, A, mu = qnm_func(chif)
        mu = mu[index]
            
        # Use symmetry properties to get the mirror mixing coefficient, if 
        # requested
        if sign == -1:
            mu = (-1)**(l+lp)*np.conjugate(mu)
            
        return mu
        
    def mu_list(self, indices, chif):
        """
        Return a list of mixing coefficients, for all requested indices.
        
        Parameters
        ----------
        indices : array_like
            A sequence of (l,m,l',m',n',sign) tuples specifying which mixing 
            coefficients to return.
            
        chif : float
            The dimensionless spin magnitude of the final black hole.
        Returns
        -------
        mus : list
            The list of spherical-spheroidal mixing coefficients.
        """
        # List to store the mixing coeffs
        mus = []
        
        # For each mode, call the qnm function and append the result to the 
        # list
        for l, m, lp, mp, nprime, sign in indices:
            mus.append(self.mu(l, m, lp, mp, nprime, sign, chif))
        
        return mus