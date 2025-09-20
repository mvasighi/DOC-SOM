function net=dbgsom_init(data)

% DBGSOM initializatio 

% Directed Batch Growing Self Organizing Map (DBGSOM)
% version 1.0 - July 2017
% Mahdi Vasighi
% Institute for Advanced Studies in Basic Sciences, Zanjan, Iran
% Department of Computer Science and Information Technology
% www.iasbs.ac.ir/~vasighi/

%     Directed Batch Growing Self Organizing Map (DBGSOM), version 1.0
%     Copyright (C) 2017  Mahdi Vasighi
% 
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     any later version.
% 
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <http://www.gnu.org/licenses/>.

grd=[0 0;1 0;0 1;1 1]'; %grid positions
W=mean(data)'*ones(1,4)+0.1*randn(size(data,2),size(grd,2));
        %                 W=rand(size(X,2),size(grd,2));
% [coeff, b, c] = pca(data);
% unit_v1 = coeff(:,1) / norm(coeff(:,1));
% unit_v2 = coeff(:,2) / norm(coeff(:,2));
% W2 = mean(data)'*ones(1,4)+0.05 *[unit_v1,-unit_v1,unit_v2,-unit_v2];
% 
% W3 = mean(b)'*ones(1,4)+0.05 *[unit_v1,-unit_v1,unit_v2,-unit_v2];

net.W=W;
net.grd=grd;