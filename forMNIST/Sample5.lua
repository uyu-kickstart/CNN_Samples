--[[
カーネルのサイズ5*5,カーネルの枚数3+2
input->(1)->(2)->(3)->(4)->(5)->output
input 1*W*W (W = 16, 24, 28, 32)
(1) 全結合層		ReLU										output:classes*3
(2) 全結合層		(Log)SoftMax								output:classes
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
mlp:add(nn.Reshape(1*width*width))
mlp:add(nn.Linear(1*width*width, 3*classes))
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
	out = mlp:forward(traindata[i][1]):exp()
	max, idx = 0
	for j=1,classes do
		if out[1][j] > max then 
			max = out[1][j]
			idx = j
		end
	end
	if idx == traindata[i][2] then
		count = count + 1
	end
end
io.write("train recognize : ")
io.write(count/traindata.size() * 100)
print(" %")

count = 0
for i=1,testdata.size() do
	out = mlp:forward(testdata[i][1]):exp()
	max,idx = 0
	for j=1,10 do
		if out[1][j] > max then 
			max = out[1][j]
			idx = j
		end
	end
	if idx == testdata[i][2] then
		count = count + 1
	end
end
io.write("test recognize : ")
io.write(count/testdata.size() * 100)
print(" %")

torch.save("model_data_5",mlp)