function [opt]=mrQ_PDfit_ParallerCoils(outDir,SunGrid,M0cfile,degrees,subName,proclass,prctileClip,numofcoupels,clobber)
%
% [opt]=mrQ_PDfit_ParallerCoils(outDir,SunGrid,M0cfile,degrees,subName,proclass,clobber)
% # Create the PD from the boxes
%
% INPUTS:
%   outDir      - The output directory - also reading file from there
%
%   SunGrid     - Flag to use the SGE for computations
%   M0cfile     - The combined/aligned M0 data
%   degrees     - Polynomial degrees for the coil estimation
%   subName     - the subject name for the SGE run
%   prctileClip - the part of the data we clip becouse it is changing too
%                  fast in space and hard to be fit by poliynomyials
%   numofcoupels - number of couple of coil each coil will be mach too
%                   (defult 3)
%  clobber:     - Overwrite existing data and reprocess. [default = false]
% OUTPUTS:
% opt           - a  structure that save the fit parameter is saved in the outdir and in
%               the tmp directory named fitLog.mat
%               calling to FitM0_sanGrid_v2 that save the output fitted files in tmp directorry
%              this will be used lster by mrQfitPD_multiCoils_M0 to make the PD map
%
%
% SEE ALSO:
%   
%
%
% ASSUMPTIONS:
%   This code assumes that your SGE output directory is '~/sgeoutput'
%
%
% (C) Stanford University, VISTA
%
%


%% CHECK INPUTS AND SET DEFAULTS

if (notDefined('outDir') || ~exist(outDir,'dir'))
    outDir = uigetDir(pwd,'Select outDir');
end

if notDefined('degrees')
    disp('Using the defult polynomials: Degrees = 3 for coil estimation');
    degrees = 4; %the input number of the function is n+1 so it is 4 and not 3...
end

%using sungrid  defult no
if(~exist('SunGrid','var'))
    disp('SGE will not be used.');
    SunGrid = false;
end


if notDefined('numofcoupels')
    numofcoupels = 3;
end
% Clobber flag. Overwrite existing fit if it already exists and redo the PD
% Gain fits
if notDefined('clobber')
    clobber = false;
end

% we set an output strcture opt that will send to the grid with all the
% relevant information for the Gain fit
opt{1}.degrees = degrees;
opt{1}.outDir  = outDir;


% Get the subject prefix for SGE job naming
if notDefined('subName')
    % This is a job name we get from the for SGE
    [~, subName] = fileparts(fileparts(fileparts(fileparts(fileparts(outDir)))));
    disp([' Subject name for lsq fit is ' subName]);
end


     sgename    = [subName '_MultiCoilM0'];
    dirname    = [outDir '/tmpSGM0' ];
%     dirDatname = [outDir '/tmpSGM0dat' num2str(j)];
%     jumpindex  = 10; %number of boxs fro each SGR run
    
    %opt{1}.dat = M0cfile{j};
    opt{1}.name = [dirname '/M0boxfit_iter'] ;
    opt{1}.date = date;
    
    opt{1}.SGE=sgename;
    % Save out a logfile with all the options used during processing
    
    %% select the coil we will work on
    BMfile = fullfile(outDir,'HeadMask.nii.gz');
    if (exist(BMfile,'file'))
        disp(['Loading brain Mask data from ' BMfile '...']);
        BM = readFileNifti(BMfile);
        BM=logical(BM.data);
    else
        error('Cannot find the file: %s', BMfile);
    end
   opt{1}.BMfile=BMfile;
    % if the M0cfile is not an input  we load the file that was made by
    % mrQ_multicoilM0.m (defult)
    if(~exist('M0cfile','var') || isempty(M0cfile))
        M0cfile = fullfile(outDir,'AligncombineCoilsM0.nii.gz');
    end
    if ~exist(M0cfile,'file')
        disp(' can not find the multi coils M0 file')
        error
    else
        M0=readFileNifti(M0cfile);
    end

coils=size(M0.data,4);
 opt{1}.dat = M0cfile;

% we need to find the coils data that is usful (couple of coils that can be
% mached). and define the area of the coil that can be fitted by
% polynoyials ((above the noise floor and not to varing to fast in space).
if notDefined('prctileClip')
    
    if coils>10; % i just guess here this need to be check with the spesipic coil in use
        prctileClip=98;
    else
        prctileClip=99;
    end
end
 opt{1}.prctileClip =prctileClip;

for i=1:coils;
    
    in=M0.data(:,:,:,i);
    up=prctile(in(BM),prctileClip); %we clip the highest SNR voxels bexouse they are imposible to fit with polynomials
    med=median(in(BM));
    mask=BM;
    mask(in<med)=0; % we clip the noise part below of the median value (maybe this need to be different for differnt coils
    mask(in>up)=0;
    M0mask(:,:,:,i)=mask; 
end
clear in up med mask

for i=1:coils;
    
    for j=1:coils
    in= M0mask(:,:,:,i)+M0mask(:,:,:,j);
    Val(j)=length(find(in==2));
    end
    Val(i)=0;
    coil2coil(i)=find(max(Val)==Val);
    in=M0.data(:,:,:,i);
    med=median(in(BM));
    in1=M0.data(:,:,:,coil2coil(i));
    med1=median(in1(BM));
    ratio=(in./med)./(in1./med1);
    M0mask(:,:,:,i)=M0mask(:,:,:,i) & ratio>0.3 ; %we clip the regions that a coil is very differnt (smaller) from the most smiliar coil to it. becouse this is a marker for area we can't fit (most of the time this is blind spot of the coil)
end
opt{1}.coil2coil=coil2coil;
clear in  med  med1 in1 ratio Val coil2coil

%%
TMfile=fullfile(outDir,'FS_tissue.nii.gz');
 if ~exist(TMfile,'file')
        disp(' can not find the multi coils segmetation file')
        error
    else
        TM=readFileNifti(TMfile);
        TM=TM.data;
    end
opt{1}.TM=TMfile;

[Poly,str] = constructpolynomialmatrix3d(size(BM),find(ones(size(BM))),degrees);

for i=1:coils;
    mask=TM ==2 & M0mask(:,:,:,i);
    [params1,gains,rs] = fit3dpolynomialmodel(M0.data(:,:,:,i),mask,degrees);
    
    % we check if there are point that are off with this model (we
    % won't use it)
    G=Poly*params1';
    G=reshape(G,size(BM));
    % up date the mask and  exclude crazy points 50% more or less then WH %
    % we also exclude point that are end up having greater value after
    % corrections
    M0mask(:,:,:,i)=M0mask(:,:,:,i) & M0.data(:,:,:,i)./G>0.5 & M0.data(:,:,:,i)./G<1.5   & M0.data(:,:,:,i)./G<M0.data(:,:,:,i);
    % re fit inital model
    mask=TM ==2 & M0mask(:,:,:,i);
    [params1,gains,rs] = fit3dpolynomialmodel(M0.data(:,:,:,i),mask,degrees);
    % up date the mask and  exclude crazy points ( we can do it more and
    % more  as a loop but maybe this is enghf.
        G=Poly*params1';
            G=reshape(G,size(BM));

    M0mask(:,:,:,i)=M0mask(:,:,:,i) & M0.data(:,:,:,i)./G>0.5 & M0.data(:,:,:,i)./G<1.5  & M0.data(:,:,:,i)./G<M0.data(:,:,:,i);
    
    %save the inital parameters
    x0(i,:)=params1;
    
end;
opt{1}.x0=x0;
clear G mask TM params1 gains rs x0
%% now we will define who fit with who

couples=zeros(coils,numofcoupels);
coilsCoples=zeros(coils,coils);

for i=1:coils;
    
    for j=1:coils
        in= M0mask(:,:,:,i)+M0mask(:,:,:,j);
        Val(j)=length(find(in==2));
        clear in
    end
    Val(i)=0;
    
    for kk=1:numofcoupels
        
        coil=find(max(Val)==Val);
        if coilsCoples(i,coil)==0
            coilsCoples(i,coil)=1;
            coilsCoples(coil,i)=-1;
            couples(i,kk)=coil;
        end
        Val(coil)=0;
        clear coil
    end
    clear Val
end
opt{1}.couples=couples;
clear  coilsCoples
%%
logname = [outDir '/PDGridCall_Log.mat'];
    
    save(logname,'opt');
    if clobber && (exist(dirname,'dir'))
        % in the case we start over and there are  old fits, so we will
        % deleat them
        eval(['! rm -r ' dirname]);
    end
    %%   Perform the gain fits
    % Perform the fits for each box using the Sun Grid Engine
    if SunGrid==1;
        
        % Check to see if there is an existing SGE job that can be
        % restarted. If not start the job, if yes prompt the user.
        if (~exist(dirname,'dir')),
            mkdir(dirname);
            eval(['!rm -f ~/sgeoutput/*' sgename '*'])
            if proclass==1
                sgerun2('mrQ_PDfit_ParallerCoils_Gridcall(logname,jobindex);',sgename,1,1:length(find(couples)),[],[],15000);

            else 
                sgerun('mrQ_PDfit_ParallerCoils_Gridcall(logname,jobindex);',sgename,1,1:length(find(couples)),[],[],15000);
                
            end
        else
            % Prompt the user
            inputstr = sprintf('An existing SGE run was found. \n Would you like to try and finish the exsist SGE run?');
            an1 = questdlg( inputstr,'mrQ_fitPD_multiCoils','Yes','No','Yes' );
            if strcmpi(an1,'yes'), an1 = 1; end
            if strcmpi(an1,'no'),  an1 = 0; end
            
            % User opted to try to finish the started SGE run
            if an1==1
                reval = [];
                list  = ls(dirname);
                ch    = 1:jumpindex:length(find(couples));
                k     = 0;
                
                for ii=1:length(ch),
                    ex=['_' num2str(ch(ii)) '_'];
                    if length(regexp(list, ex))==0,
                        k=k+1;
                        reval(k)=(ii);
                    end
                end
                
                if length(find(reval)) > 0
                    eval(['!rm -f ~/sgeoutput/*' sgename '*'])
                    if proclass==1
                        for kk=1:length(reval)
                            sgerun2('mrQ_PDfit_ParallerCoils_Gridcall(logname,jobindex);',[sgename num2str(kk)],1,reval(kk),[],[],15000);

                        end
                    else
                            sgerun('mrQ_PDfit_ParallerCoils_Gridcall(logname,jobindex);',sgename ,1,reval,[],[],15000);
                    end
                end
                
                % User opted to restart the existing SGE run
            elseif an1==0,
                t = pwd;
                cd (outDir)
                eval(['!rm -rf ' dirname]);
                cd (t);
                eval(['!rm -f ~/sgeoutput/*' sgename '*'])
                mkdir(dirname);
                if proclass==1
                sgerun2('mrQ_PDfit_ParallerCoils_Gridcall(logname,jobindex);',sgename,1,1:length(find(couples)),[],[],15000);
                else
                sgerun('mrQ_PDfit_ParallerCoils_Gridcall(logname,jobindex);',sgename,1,1:length(find(couples)),[],[],15000);
                    
                end
            else
                error('User cancelled');
            end
        end
        
    end
    

