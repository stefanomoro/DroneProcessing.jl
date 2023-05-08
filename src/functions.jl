#import ArchGDAL as AG
#import GeoInterface
using AstroTime  # Base.DateTime only handles milliseconds
#using Base.Iterators: flatten
using DSP: amp2db, pow2db, unwrap, kaiser, hamming
#using FFTW; FFTW.set_num_threads(Threads.nthreads())
using FIRInterp # https://github.jpl.nasa.gov/bhawkins/FIRInterp.jl
#using GDAL: gdalsetmetadata
using Geodesy
#using JLD2
using LinearAlgebra
using Optim
#using Plots: plot
#using PyPlot: imshow
using Polynomials
using StaticArrays
using Statistics: mean, median, std
using StatsBase
using Interpolations
using ImageFiltering

# some utilities for static arrays
const Vec3 = SVector{3, Float64}
const Mat3 = SMatrix{3, 3, Float64}
const Vec3List = AbstractArray{Vec3}

const c = 299_792_458 # m/s

# copy
# component(vecs::Vec3List, i) = [v[i] for v in vecs]
# view
component(vecs::Vec3List, i) = @view(reinterpret(reshape, eltype(Vec3), vecs)[i,:])
getx(vecs) = component(vecs, 1)
gety(vecs) = component(vecs, 2)
getz(vecs) = component(vecs, 3)

function compute_ned2sch(pos_ned)
    A = [getx(pos_ned[:]) gety(pos_ned[:])]
    origin = mean(A, dims=1)
    A .-= origin
    B = svd(A'A)
    # Transforming from NED to SCH with SVD. 
    # The transformation is done with Vt (V hermitian), knowing that A = U * diag(S) * Vt
    ned2sch = Mat3([
        B.Vt[1,1] B.Vt[1,2] 0;
        B.Vt[2,1] B.Vt[2,2] 0;
        0 0 -1
    ])
    return ned2sch
end

function mov_mean(vec,avg_len::Int)
    return imfilter(vec, ones(1,avg_len)./avg_len)
end

function movavg_drone_pos!(x_tx::Array{Vec3},x_rx::Array{Vec3},avg_len::Int)
    tx_x = mov_mean(getx(x_tx)',avg_len)'
    tx_y = mov_mean(gety(x_tx)',avg_len)'
    tx_z = mov_mean(getz(x_tx)',avg_len)'
    rx_x = mov_mean(getx(x_rx)',avg_len)'
    rx_y= mov_mean(gety(x_rx)',avg_len)'
    rx_z = mov_mean(getz(x_rx)',avg_len)'

    for i = 1:length(tx_x)
        x_tx[i] = Vec3(tx_x[i], tx_y[i], tx_z[i])
        x_rx[i] = Vec3(rx_x[i], rx_y[i], rx_z[i])
    end
    return x_tx,x_rx
end

function polyfit_drone_pos(t,x_tx::Vec3List,x_rx::Vec3List,pol_ord::Int)

    fit_tx_x = Polynomials.fit(t, getx(x_tx), pol_ord)
    fit_tx_y = Polynomials.fit(t, gety(x_tx), pol_ord)
    fit_tx_z = Polynomials.fit(t, getz(x_tx), pol_ord)

    fit_rx_x = Polynomials.fit(t, getx(x_rx), pol_ord)
    fit_rx_y = Polynomials.fit(t, gety(x_rx), pol_ord)
    fit_rx_z = Polynomials.fit(t, getz(x_rx), pol_ord)

    x_tx_pol = copy(x_tx)
    x_rx_pol = copy(x_rx)
    for i = 1:length(x_tx)
        x_tx_pol[i] = Vec3(fit_tx_x(t[i]), fit_tx_y(t[i]), fit_tx_z(t[i]))
        x_rx_pol[i] = Vec3(fit_rx_x(t[i]), fit_rx_y(t[i]), fit_rx_z(t[i]))
    end
    return x_tx_pol,x_rx_pol
end

# Compute bistatic range to CR at pulse times.
function bistatic_range(x_tx::Vec3List, x_rx::Vec3List, target)
    n = length(x_tx)
    @assert length(x_rx) == n
    r = zeros(n)
    for i = 1:n
        r[i] = (norm(x_tx[i] - target) + norm(x_rx[i] - target)) / 2
    end
    return r
end

# nearest neighbor
function find_peak_near(t::U, z::AbstractVector{Complex{T}}; window=30) where {T, U<:Number}
    w2 = round(Int, (window - 1) / 2)
    i = round(Int, t)
    i0 = max(1, i - w2)
    i1 = min(length(z), i + w2)
    chunk = @view(z[i0:i1])
    k = argmax(abs2.(chunk)) + i0 - 1
    return U(k), z[k]
end

# interpolated
function find_peak_near(t, z::AbstractVector{Complex{T}}, itp; window=30) where {T}
    k, zk = find_peak_near(t, z; window=window) # get nearest neighbor without interp
    t0 = k - window / 2
    t1 = t0 + window
    f(t) = -abs2(interp(itp, z, t))
    result = optimize(f, t0, t1)
    tmax = result.minimizer
    return tmax, interp(itp, z, tmax)
end
function compute_roff_shift(cr_shifts,nbins)
    bin_size = (minimum(cr_shifts)-maximum(cr_shifts))/nbins
    bins = range(minimum(cr_shifts),stop=maximum(cr_shifts),length=nbins)
    hist = StatsBase.fit(Histogram,cr_shifts,bins)
    idx = argmax(hist.weights)
    r_off_mode = (hist.edges[1][idx] + hist.edges[1][idx+1])/2
    return r_off_mode , hist
end

function wrap(phase)
    return mod(phase + π, 2π) - π
end

function phase_detrending(ppeak, kω, r_cr , t, mask_cr)
    # we sum because the radar data phase is physically consider to rotate with minus, 
    # so pppeak = -kω * range

    ppeak_detrend = zero(ppeak)
    for i=1:length(ppeak)
        ppeak_detrend[i] = wrap(ppeak[i] + kω * r_cr[i])
    end
    
    unwpeak_detrend = unwrap(ppeak_detrend) / kω
    linear_fit = Polynomials.fit(t[mask_cr], unwpeak_detrend[mask_cr], 1)
    unwpeak_detrend_nolin = unwpeak_detrend .- linear_fit.(t)
    
    return ppeak_detrend, unwpeak_detrend, unwpeak_detrend_nolin
end

# For complex data detect power first.  Otherwise just average.
# Is an averaging filter, to sub sample the image over the s direction
function azlook(z::AbstractArray{Complex{T},2}, nl) where {T}
    nr, ns = size(z)
    nso = floor(Int, ns / nl)
    zout = zeros(T, nr, nso)
    Threads.@threads for i=1:nso
        @inbounds for j=1:nl
            k = (i - 1) * nl + j
            zout[:,i] .+= abs2.(z[:,k])
        end
    end
    return zout ./ nl 
end

function azlook(x::AbstractArray{T,2}, nl) where {T <: Real}
    nr, ns = size(x)
    nso = floor(Int, ns / nl)
    xout = zeros(T, nr, nso)
    Threads.@threads for i=1:nso
        @inbounds for j=1:nl
            k = (i - 1) * nl + j
            xout[:,i] .+= x[:,k]
        end
    end
    return xout ./ nl
end

# For applying the same filter to azimuth axis for plotting.
function look1d(x, nl)
    n = length(x)
    no = floor(Int, n / nl)
    out = zeros(eltype(x), no)
    for i=1:no, j=1:nl
        k = (i - 1) * nl + j
        out[i] += x[k]
    end
    return out ./ nl
end

# use hamming window to de-spckle. No decimation
function hamming_despeckle(z::Matrix{T}, nl::Integer, skipEqualize = false) where T <: Complex
    nr, ns = size(z)
    zout = zeros(Float64, nr, ns)
    for i=1:nr
        zout[i,:] = imfilter(abs2.(z[i,:]),hamming(nl))
    end
    zout ./= nl 
    zout = pow2db.(zout)
    if skipEqualize
        return zout
    end
    return zout .- maximum(zout)
end

function  gaussActivFunc(x,sigma)
    #GAUSSACTIVFUNC gaussian activation function
    #   [out] = gaussActivFunc(x,sigma)
    exp(-0.5*((x)./sigma).^2)
end


function backproject_stripmap_sch(itp, z, r0, dr, kω, href, beamwidth, x_bp::Vec3List, x_tx::Vec3List, x_rx::Vec3List)     
    npulse = size(z, 2)
    @assert npulse == length(x_tx) == length(x_rx)
    out = zeros(eltype(z), size(x_bp))
    #compute the s coordinate in the center between tx-rx for each pulse
    s_pulse = (getx(x_tx) .+ getx(x_rx)) ./ 2
    Threads.@threads for i in eachindex(x_bp)
        s, cross_t, h = x_bp[i]
        range_zerodop = sqrt(cross_t^2 + (href - h)^2)
        synth_aperture_size = range_zerodop * beamwidth
        s_start = s - synth_aperture_size / 2
        s_stop =  s + synth_aperture_size / 2
        @inbounds for j = 1:npulse
            # using a rectangle to limit the beamwidth, so the wavenumber bandwidth
            if s_start <= s_pulse[j] <= s_stop
                r = (norm(x_bp[i] - x_tx[j]) + norm(x_bp[i] - x_rx[j])) / 2
                ir = 1 + (r - r0) / dr
                out[i] += cis(kω * (r - range_zerodop)) * interp(itp, @view(z[:,j]), ir)
            end
        end
    end
    return out
end

function TDBP_wavenumber(itp, z, R0, dR, λ, az_res, squint_ang, x_bp::Vec3List, x_tx::Vec3List, x_rx::Vec3List)
    npulse = size(z,2)
    @assert npulse == length(x_tx) == length(x_rx) "Slow time length not matching"
    out = zeros(eltype(z), size(x_bp))
    Δk =  2π / az_res
    k_0 = 2 * sin(squint_ang) * 2π/λ
    Threads.@threads for i in eachindex(x_bp)
        @inbounds for j in 1:npulse
            Rtx = norm(x_bp[i] - x_tx[j]) 
            Rrx = norm(x_bp[i] - x_rx[j])

            # get ψ angle of the planar
            sinψtx = (x_tx[j][1] - x_bp[i][1]) / Rtx
            sinψrx = (x_rx[j][1] - x_bp[i][1]) / Rrx
            k = (sinψtx +sinψrx) * 2π/λ 
            Wn = gaussActivFunc(k - k_0, Δk/2)
            if Wn <= .1
                continue
            end
            iR =  1 + ((Rtx + Rrx)/2 - R0) /dR
            out[i] += Wn * interp(itp,@view(z[:,j]),iR) * cis(2π * (Rtx+Rrx)/λ  )
        end
    end
    return out
end

function findMaxPow(multi_squint::Array{Complex, 3})
    maxV = -Inf
    for i in 1:size(multi_squint,3)
        tempM = maximum(abs2.(multi_squint[:,:,i]))
        if tempM > maxV
            maxV = tempM
        end
    end
    return maxV
end
