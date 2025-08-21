import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "Comfy.GeminiStructuredOutput",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "GeminiStructuredOutput") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const ret = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                const schemaWidget = this.widgets.find(w => w.name === "schema_json");
                const enumWidget = this.widgets.find(w => w.name === "enum_options");
                const modeWidget = this.widgets.find(w => w.name === "output_mode");
                
                if (schemaWidget) {
                    schemaWidget.inputEl.style.fontFamily = "monospace";
                    schemaWidget.inputEl.style.fontSize = "12px";
                    schemaWidget.inputEl.rows = 10;
                    
                    schemaWidget.inputEl.addEventListener("input", function() {
                        try {
                            JSON.parse(this.value);
                            this.style.borderColor = "#4CAF50";
                            this.style.backgroundColor = "rgba(76, 175, 80, 0.05)";
                        } catch (e) {
                            this.style.borderColor = "#f44336";
                            this.style.backgroundColor = "rgba(244, 67, 54, 0.05)";
                        }
                    });
                }
                
                if (enumWidget) {
                    enumWidget.inputEl.style.fontFamily = "monospace";
                    enumWidget.inputEl.style.fontSize = "12px";
                    enumWidget.inputEl.rows = 5;
                    
                    enumWidget.inputEl.addEventListener("input", function() {
                        try {
                            const parsed = JSON.parse(this.value);
                            if (Array.isArray(parsed)) {
                                this.style.borderColor = "#4CAF50";
                                this.style.backgroundColor = "rgba(76, 175, 80, 0.05)";
                            } else {
                                throw new Error("Not an array");
                            }
                        } catch (e) {
                            this.style.borderColor = "#f44336";
                            this.style.backgroundColor = "rgba(244, 67, 54, 0.05)";
                        }
                    });
                }
                
                const updateVisibility = () => {
                    if (modeWidget) {
                        const mode = modeWidget.value;
                        if (schemaWidget) {
                            schemaWidget.inputEl.style.display = mode === "json_schema" ? "block" : "none";
                            const label = schemaWidget.inputEl.previousElementSibling;
                            if (label) label.style.display = mode === "json_schema" ? "block" : "none";
                        }
                        if (enumWidget) {
                            enumWidget.inputEl.style.display = mode === "enum" ? "block" : "none";
                            const label = enumWidget.inputEl.previousElementSibling;
                            if (label) label.style.display = mode === "enum" ? "block" : "none";
                        }
                    }
                };
                
                if (modeWidget) {
                    modeWidget.callback = () => {
                        updateVisibility();
                    };
                }
                
                setTimeout(updateVisibility, 100);
                
                const addSchemaTemplateButton = () => {
                    const button = document.createElement("button");
                    button.textContent = "📋 Schema Templates";
                    button.style.marginTop = "5px";
                    button.style.padding = "5px 10px";
                    button.style.fontSize = "12px";
                    button.style.cursor = "pointer";
                    
                    const templates = {
                        "User Profile": {
                            type: "object",
                            properties: {
                                name: { type: "string" },
                                age: { type: "integer", minimum: 0 },
                                email: { type: "string", format: "email" },
                                bio: { type: "string", maxLength: 500 },
                                tags: { type: "array", items: { type: "string" } }
                            },
                            required: ["name", "email"]
                        },
                        "Product Info": {
                            type: "object",
                            properties: {
                                product_name: { type: "string" },
                                price: { type: "number", minimum: 0 },
                                category: { type: "string" },
                                in_stock: { type: "boolean" },
                                specifications: {
                                    type: "object",
                                    properties: {
                                        weight: { type: "number" },
                                        dimensions: { type: "string" }
                                    }
                                }
                            },
                            required: ["product_name", "price"]
                        },
                        "Article Summary": {
                            type: "object",
                            properties: {
                                title: { type: "string" },
                                summary: { type: "string", maxLength: 200 },
                                key_points: {
                                    type: "array",
                                    items: { type: "string" }
                                },
                                sentiment: {
                                    type: "string",
                                    enum: ["positive", "neutral", "negative"]
                                },
                                confidence: {
                                    type: "number",
                                    minimum: 0,
                                    maximum: 1
                                }
                            },
                            required: ["title", "summary"]
                        }
                    };
                    
                    button.onclick = () => {
                        const select = document.createElement("select");
                        select.innerHTML = '<option value="">Select a template...</option>';
                        
                        Object.keys(templates).forEach(name => {
                            const option = document.createElement("option");
                            option.value = name;
                            option.textContent = name;
                            select.appendChild(option);
                        });
                        
                        select.onchange = () => {
                            if (select.value && schemaWidget) {
                                schemaWidget.value = JSON.stringify(templates[select.value], null, 2);
                                schemaWidget.inputEl.value = schemaWidget.value;
                                schemaWidget.inputEl.dispatchEvent(new Event("input"));
                            }
                        };
                        
                        if (button.parentNode.querySelector("select")) {
                            button.parentNode.querySelector("select").remove();
                        } else {
                            button.parentNode.insertBefore(select, button.nextSibling);
                        }
                    };
                    
                    if (schemaWidget && schemaWidget.inputEl.parentNode) {
                        schemaWidget.inputEl.parentNode.appendChild(button);
                    }
                };
                
                setTimeout(addSchemaTemplateButton, 200);
                
                return ret;
            };
        }
        
        if (nodeData.name === "GeminiJSONExtractor") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const ret = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                const fieldsWidget = this.widgets.find(w => w.name === "extract_fields");
                
                if (fieldsWidget) {
                    fieldsWidget.inputEl.style.fontFamily = "monospace";
                    fieldsWidget.inputEl.style.fontSize = "12px";
                    fieldsWidget.inputEl.rows = 8;
                    
                    const addFieldTemplateButton = () => {
                        const button = document.createElement("button");
                        button.textContent = "📝 Field Templates";
                        button.style.marginTop = "5px";
                        button.style.padding = "5px 10px";
                        button.style.fontSize = "12px";
                        button.style.cursor = "pointer";
                        
                        const templates = {
                            "Article Analysis": "title: string\nauthor: string\npublish_date: string\ncategory: string\nsummary: string\nkey_points: string[]\nsentiment: string",
                            "Person Info": "name: string\nage: integer\noccupation: string\nlocation: string\nskills: string[]",
                            "Event Details": "event_name: string\ndate: string\ntime: string\nlocation: string\nattendees: integer\ndescription: string",
                            "Product Review": "product_name: string\nrating: number\npros: string[]\ncons: string[]\nrecommendation: boolean\nsummary: string"
                        };
                        
                        button.onclick = () => {
                            const select = document.createElement("select");
                            select.innerHTML = '<option value="">Select a template...</option>';
                            
                            Object.keys(templates).forEach(name => {
                                const option = document.createElement("option");
                                option.value = name;
                                option.textContent = name;
                                select.appendChild(option);
                            });
                            
                            select.onchange = () => {
                                if (select.value && fieldsWidget) {
                                    fieldsWidget.value = templates[select.value];
                                    fieldsWidget.inputEl.value = fieldsWidget.value;
                                }
                            };
                            
                            if (button.parentNode.querySelector("select")) {
                                button.parentNode.querySelector("select").remove();
                            } else {
                                button.parentNode.insertBefore(select, button.nextSibling);
                            }
                        };
                        
                        if (fieldsWidget && fieldsWidget.inputEl.parentNode) {
                            fieldsWidget.inputEl.parentNode.appendChild(button);
                        }
                    };
                    
                    setTimeout(addFieldTemplateButton, 200);
                }
                
                return ret;
            };
        }
    }
});