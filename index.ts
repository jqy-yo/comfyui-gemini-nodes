import { GoogleGenerativeAI, SchemaType } from '@google/generative-ai';


const t = async () => {
    const genAI = new GoogleGenerativeAI("AIzaSyAU2SWpQv8zelzm0ehANJATQmsjYv2CDbs");

    // Define the structured schema based on the provided JSON structure.
    const weatherSchema = {
        type: SchemaType.OBJECT,
        properties: {
            description: { type: SchemaType.STRING },
        },
        required: [
            "description"
        ]
    };

    const weatherModel = genAI.getGenerativeModel({
        model: "gemini-2.0-flash",
        generationConfig: {
            responseMimeType: "application/json",
            responseSchema: weatherSchema,
            temperature: 0.7,
            topP: 1,
            maxOutputTokens: 8192,
        },
    });
    const prompt = `请根据**标题**和**对话内容**生成用于flux模型图像生成场景的提示词，生成的图像用于记忆英语单词的配图，帮助用户加深记忆，要求：
**着重参考标题内容**
输出要求：
**只输出提示词，禁止一切无关内容，禁止废话**
*提示词中禁止出现也许、或、比如等不确定的词语*
构图要求：
**必须体现重点事物，可以使用幻想气泡等任何方式突出表现**
**判断符合人物场景或者纯景场景**
**天气、建筑、特殊物品等需要使用纯景场景**
1、人物场景：
**人种：欧美人
焦点：人物的表情和肢体语言要符合对话
背景：展现对话主题相关的场景**
2、纯景场景：
**严格生成符合对话内容的场景的提示词**

风格要求：
皮克斯风格

提示词格式：
英文提示词

根据对话内容确定：
背景环境
时间和光线
整体氛围

对话内容以及标题：
[lightness]
Why do you feel such lightness in your heart today?
I finally finished my exams, and it feels amazing!`
    try {
        for (let i = 0; i < 10; i++) {
            console.group('========')
            const result = await weatherModel.generateContent(prompt);
            const responseText = result.response.text();
            // console.log("result", responseText)
            const forecastData = JSON.parse(responseText);
            console.log(forecastData);
            console.group('========')
        }

    } catch (err) {
        console.error("Failed to generate weather forecast:", err);

    }
}


t();