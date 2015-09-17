--[[
カーネルのサイズ5*5,カーネルの枚数3+2
input->(1)->(2)->(3)->(4)->(5)->output
input 1*W*W (W = 16, 24, 28, 32)
(1) たたみ込み	ReLU	filter:4	karnel:7*7 	zeroPad:なし	output:4*(W-6)*(W-6)
(2) たたみ込み	ReLU	filter:3	karnel:5*5	zeroPad:なし	output:12*(W-10)*(W-10)
(3) Maxプーリング						karnel:3*3	zeroPad:なし	output:12*(W-12)*(W-12)
(4) 全結合層		ReLU										output:classes*3
(5) 全結合層		(Log)SoftMax								output:classes
誤差関数:交差エントロピー
]]

local mnist = require "mnist"

local trainset = mnist.traindataset()
local testset = mnist.testdataset()
--local debug_input = torch.Tensor(1,28,28):uniform()

local traindata = {}
local data = trainset.data:double() / 255
local N = 1000
function traindata.size()
	return N
end
for i = 1, N do
	traindata[i] = {data[i]:reshape(1,28,28), trainset.label[i]+1}
end

local testdata = {}
local data_test = testset.data:double() / 255
local M = 1000
function testdata.size()
	return M
end
for i = 1, M do
	testdata[i] = {data_test[i]:reshape(1,28,28), testset.label[i]+1}
end

local nn = require "nn"

width = 28		--入力画像の辺の長さ
classes = 10	--分類クラス数

mlp = nn.Sequential()
mlp:add(nn.SpatialConvolutionMM(1, 1*4, 7, 7,1,1,0,0))	--(channel_in, channel_out, kW, kH)
mlp:add(nn.ReLU())
--mlp:add(nn.Dropout(0.2))
mlp:add(nn.SpatialConvolutionMM(4, 3*4, 3, 3,1,1,0,0))
mlp:add(nn.ReLU())
--mlp:add(nn.Dropout(0.3))
mlp:add(nn.SpatialMaxPooling(3,3,1,1,0,0))	--(kW, kH)
mlp:add(nn.Reshape(3 * 4 * (width-12) * (width-12)))
mlp:add(nn.Linear(3 * 4 * (width-12) * (width-12), 3*classes))
mlp:add(nn.ReLU())
--mlp:add(nn.Dropout(0.5))
mlp:add(nn.Linear(3*classes, classes))
mlp:add(nn.LogSoftMax())

print(mlp)

criterion = nn.ClassNLLCriterion()

trainer = nn.StochasticGradient(mlp, criterion)
trainer.maxIteration = 50
trainer.learningRate = 0.005
trainer:train(traindata)

count = 0
for i=1,traindata.size() do
	max,idx = mlp:forward(traindata[i][1]):exp():max(1)
	if idx[1] == traindata[i][2] then
		count = count + 1
	end
end
io.write("train recognize : ")
io.write(count/traindata.size() * 100)
print(" %")

count = 0
for i=1,testdata.size() do
	max,idx = mlp:forward(testdata[i][1]):exp():max(1)
	if idx[1] == testdata[i][2] then
		count = count + 1
	end
end
io.write("test recognize : ")
io.write(count/testdata.size() * 100)
print(" %")

torch.save("model_data_1_4",mlp)