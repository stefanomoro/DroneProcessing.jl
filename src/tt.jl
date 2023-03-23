Ns = size(x_bp,1);
Nc = size(x_bp,2);
t = r ./ c;

# Processed Antenna aperture
psi_proc = λ / 2 / scenario.grid.pho_az;
# Processed wavenumbers
Δk = 2π /ρ_az;

# Squint angle vectors
squint_angles = [0];

# Copy variables for optimizing parfor
#reset(gpuDevice)
idxs = t >= 0;
t = t(idxs);
RC = radar.RC(idxs,:);
# remove mean value from grid 
X = scenario.grid.X;
Y = scenario.grid.Y;
reference_x = mean(X(:,1));
reference_y = mean(Y(1,:));
X = single(X - reference_x);
Y = single(Y - reference_y);
ref = [reference_x;reference_y;0];

TX_pos = single(TX.pos - ref);
TX_pos_x = gpuArray(TX_pos(1,:));TX_pos_y = gpuArray(TX_pos(2,:));TX_pos_z = gpuArray(TX_pos(3,:)); 
RX_pos = single(RX.pos - ref);
RX_pos_x = gpuArray(RX_pos(1,:));RX_pos_y = gpuArray(RX_pos(2,:));RX_pos_z = gpuArray(RX_pos(3,:)); 
RX_speed = gpuArray(single(RX.speed));
X = gpuArray(X); Y = gpuArray(Y); z0 = single(scenario.grid.z0);
lambda = single(param.lambda); f0 = single(param.f0);
RC = gpuArray(single(RC));
x_ax = gpuArray(single(scenario.grid.x_ax));
t = gpuArray(single(t));
median_speed = median(RX_speed);


# Initialize vectors for the result
focus.Focused_vec = zeros(size(X,1),size(X,2),length(focus.angle_vec),'single');
# focus.not_coh_sum = zeros(size(focus.Focused_vec),'single');
focus.SumCount = zeros(size(focus.Focused_vec),'single');

tic
for ang_idx = 1:length(focus.angle_vec)
    waitbar(ang_idx/length(focus.angle_vec),wbar,strcat("Backprojecting n "...
        ,num2str(ang_idx),"/",num2str(length(focus.angle_vec))));
    
    psi_foc = deg2rad(focus.angle_vec(ang_idx));
    k_rx_0 = single(sin(psi_foc).*(2*pi/param.lambda)); 
    
    S = gpuArray(zeros(Nx,Ny,'single'));
#     A = zeros(Nx,Ny,'gpuArray');
    SumCount = gpuArray(zeros(Nx,Ny,'single'));
    parfor n = 1 : size(RC,2)
        [Sn,Wn] = elementFuncTDBP(X,Y,z0,TX_pos_x(n),TX_pos_y(n),TX_pos_z(n),RX_pos_x(n),...
            RX_pos_y(n),RX_pos_z(n),lambda,Dk,RC(:,n),t,f0,k_rx_0,x_ax);
        
        
        # Give less weight to not moving positions
        speed_norm = RX_speed(n)/median_speed;
        # Count number of summations for each pixel
        SumCount = SumCount + speed_norm.*Wn;
        
        # Coherent sum over all positions along the trajectory 
        S = S + speed_norm .* Sn;
        # Inchoerent sum over all positions along the trajectory
#         A = A + abs(Sn);
    end
    waitbar(ang_idx/length(focus.angle_vec),wbar);
    
    focus.SumCount(:,:,ang_idx) = gather(SumCount);
    focus.Focused_vec(:,:,ang_idx) = gather(S);
#     focus.not_coh_sum(:,:,ang_idx) = gather(A); 
end

close(wbar)

disp (strcat("Total elaboration time: ",num2str(toc/60)," min"))
